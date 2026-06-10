from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _make_activation(name: str) -> nn.Module:
    activation_name = name.lower()
    if activation_name == "elu":
        return nn.ELU(inplace=True)
    if activation_name == "relu":
        return nn.ReLU(inplace=True)
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if activation_name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# Quaternion utilities (standalone, no IsaacLab dependency)
# ---------------------------------------------------------------------------

def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion [w, x, y, z]."""
    out = q.clone()
    out[..., 1:] *= -1.0
    return out


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q ([w, x, y, z])."""
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q_w * uv + uuv)


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by inverse of quaternion q."""
    return _quat_rotate(_quat_conjugate(q), v)


def _quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Axis-angle to quaternion [w, x, y, z]."""
    sin_half = torch.sin(angle / 2.0)
    w = torch.cos(angle / 2.0)
    x = axis[..., 0] * sin_half
    y = axis[..., 1] * sin_half
    z = axis[..., 2] * sin_half
    return torch.stack([w, x, y, z], dim=-1)


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------

def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "elu",
    spectral_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims):
        linear = nn.Linear(last_dim, hidden_dim)
        if spectral_norm:
            linear = nn.utils.spectral_norm(linear)
        layers.append(linear)
        layers.append(_make_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    final_linear = nn.Linear(last_dim, output_dim)
    if spectral_norm:
        final_linear = nn.utils.spectral_norm(final_linear)
    layers.append(final_linear)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Enriched AMP state / transition builders
# ---------------------------------------------------------------------------

# Gravity vector in world frame (unit direction)
_GRAVITY_VEC_W = None


def _gravity_vec_w(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    global _GRAVITY_VEC_W
    if _GRAVITY_VEC_W is None or _GRAVITY_VEC_W.device != device or _GRAVITY_VEC_W.dtype != dtype:
        _GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
    return _GRAVITY_VEC_W


def build_amp_state(
    root_pos: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    root_quat: torch.Tensor | None = None,
    root_lin_vel: torch.Tensor | None = None,
    root_ang_vel: torch.Tensor | None = None,
    body_pos_w: torch.Tensor | None = None,
    root_pos_w: torch.Tensor | None = None,
    foot_body_ids: list[int] | None = None,
    hand_body_ids: list[int] | None = None,
) -> torch.Tensor:
    """Build enriched AMP state vector.

    Basic (current): [base_height, dof_pos, dof_vel]  ->  1 + N + N
    Enriched:        [base_height, dof_pos, dof_vel, proj_gravity(3),
                      root_lin_vel(3), root_ang_vel(3),
                      foot_pos_b(3*len(foot_body_ids)),
                      hand_pos_b(3*len(hand_body_ids))]

    When optional tensors are None, falls back to the basic representation.
    """
    if root_pos.shape[-1] >= 3:
        base_height = root_pos[..., 2:3]
    else:
        base_height = root_pos[..., :1]

    parts = [base_height, dof_pos, dof_vel]

    # --- optional enriched features ---
    if root_quat is not None:
        gravity_w = _gravity_vec_w(root_quat.device, root_quat.dtype).expand(root_quat.shape[0], 3)
        proj_gravity = _quat_rotate_inverse(root_quat, gravity_w)
        parts.append(proj_gravity)

    if root_lin_vel is not None:
        parts.append(root_lin_vel)
    if root_ang_vel is not None:
        parts.append(root_ang_vel)

    # Foot and hand positions relative to root, expressed in root frame
    if body_pos_w is not None and root_pos_w is not None and root_quat is not None:
        all_ids = []
        if foot_body_ids:
            all_ids.extend(foot_body_ids)
        if hand_body_ids:
            all_ids.extend(hand_body_ids)
        if all_ids:
            rel_pos_w = body_pos_w[:, all_ids] - root_pos_w[:, None, :]
            rel_pos_b = _quat_rotate_inverse(
                root_quat[:, None, :].expand(-1, len(all_ids), 4).reshape(-1, 4),
                rel_pos_w.reshape(-1, 3),
            ).reshape(-1, len(all_ids) * 3)
            parts.append(rel_pos_b)

    return torch.cat(parts, dim=-1)


def build_amp_transition(current_state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    return torch.cat((current_state, next_state), dim=-1)


# ---------------------------------------------------------------------------
# AMP Discriminator (enhanced)
# ---------------------------------------------------------------------------

class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 256, 256, 128),
        activation: str = "gelu",
        spectral_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.network = build_mlp(
            self.state_dim * 2,
            hidden_dims,
            1,
            activation=activation,
            spectral_norm=spectral_norm,
            dropout=dropout,
        )

    def forward(self, transition: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(transition, (tuple, list)):
            transition = build_amp_transition(transition[0], transition[1])
        logits = self.network(transition)
        return logits.squeeze(-1)

    def score(self, transition: torch.Tensor | Sequence[torch.Tensor], detach: bool = False) -> torch.Tensor:
        score = torch.sigmoid(self.forward(transition))
        return score.detach() if detach else score

    def features(self, transition: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        """Penultimate features (manual traversal); for expert stat tracking."""
        if isinstance(transition, (tuple, list)):
            transition = build_amp_transition(transition[0], transition[1])
        modules = list(self.network.children())
        x = transition
        for m in modules[:-1]:
            x = m(x)
        return x

    def score_with_features(
        self, transition: torch.Tensor | Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single traversal returning (score, penultimate_features)."""
        if isinstance(transition, (tuple, list)):
            transition = build_amp_transition(transition[0], transition[1])
        modules = list(self.network.children())
        x = transition
        for m in modules[:-1]:
            x = m(x)
        feats = x
        logits = modules[-1](x).squeeze(-1)
        return torch.sigmoid(logits), feats


class MultiScaleAMPDiscriminator(nn.Module):
    """Multi-scale AMP discriminator with separate sub-discriminators at different temporal scales."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "gelu",
        spectral_norm: bool = True,
        dropout: float = 0.1,
        num_scales: int = 2,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            AMPDiscriminator(
                state_dim=state_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                spectral_norm=spectral_norm,
                dropout=dropout,
            )
            for _ in range(num_scales)
        ])

    def forward(self, transition: torch.Tensor | Sequence[torch.Tensor], scale: int = 0) -> torch.Tensor:
        return self.discriminators[scale](transition)

    def features(self, transition: torch.Tensor | Sequence[torch.Tensor], scale: int | None = None) -> torch.Tensor | list[torch.Tensor]:
        """Return penultimate features from one or all sub-discriminators."""
        if scale is not None:
            return self.discriminators[scale].features(transition)
        return [d.features(transition) for d in self.discriminators]

    def forward_all(self, transitions: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through all discriminators.

        Args:
            transitions: list of transitions, one per scale [T1, T2, ...]
        Returns:
            list of logits, one per scale
        """
        return [disc(t) for disc, t in zip(self.discriminators, transitions)]

    def score(self, transition: torch.Tensor | Sequence[torch.Tensor], scale: int = 0, detach: bool = False) -> torch.Tensor:
        return self.discriminators[scale].score(transition, detach=detach)

    def score_all(self, transitions: Sequence[torch.Tensor], detach: bool = False) -> list[torch.Tensor]:
        return [disc.score(t, detach=detach) for disc, t in zip(self.discriminators, transitions)]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def amp_discriminator_loss(
    discriminator: AMPDiscriminator | MultiScaleAMPDiscriminator,
    expert_transition: torch.Tensor | Sequence[torch.Tensor],
    policy_transition: torch.Tensor | Sequence[torch.Tensor],
    loss_type: str = "bce",
) -> torch.Tensor:
    """Compute discriminator loss. Supports BCE and LSGAN."""
    expert_logits = discriminator(expert_transition)
    policy_logits = discriminator(policy_transition)

    if loss_type == "bce":
        expert_labels = torch.ones_like(expert_logits)
        policy_labels = torch.zeros_like(policy_logits)
        logits = torch.cat((expert_logits, policy_logits), dim=0)
        labels = torch.cat((expert_labels, policy_labels), dim=0)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    if loss_type == "lsgan":
        expert_loss = torch.mean((expert_logits - 1.0) ** 2)
        policy_loss = torch.mean(policy_logits**2)
        return 0.5 * (expert_loss + policy_loss)

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def r1_gradient_penalty(
    discriminator: AMPDiscriminator | MultiScaleAMPDiscriminator,
    expert_transition: torch.Tensor,
) -> torch.Tensor:
    """R1 gradient penalty: encourages smooth discriminator around real data."""
    expert_transition = expert_transition.detach().requires_grad_(True)
    logits = discriminator(expert_transition)
    grads = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=expert_transition,
        create_graph=True,
    )[0]
    return grads.pow(2).sum(dim=-1).mean()


def wgan_gradient_penalty(
    discriminator: AMPDiscriminator | MultiScaleAMPDiscriminator,
    expert_transition: torch.Tensor,
    policy_transition: torch.Tensor,
) -> torch.Tensor:
    """WGAN-GP: gradient penalty on random interpolations."""
    batch_size = expert_transition.shape[0]
    eps = torch.rand(batch_size, 1, device=expert_transition.device, dtype=expert_transition.dtype)
    interp = eps * expert_transition + (1 - eps) * policy_transition
    interp.requires_grad_(True)
    logits = discriminator(interp)
    grads = torch.autograd.grad(outputs=logits.sum(), inputs=interp, create_graph=True)[0]
    grads = grads.view(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def amp_discriminator_accuracy(
    discriminator: AMPDiscriminator | MultiScaleAMPDiscriminator,
    expert_transition: torch.Tensor | Sequence[torch.Tensor],
    policy_transition: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[float, float]:
    """Return (expert_accuracy, policy_accuracy) as scalar floats."""
    with torch.no_grad():
        expert_score = discriminator.score(expert_transition, detach=True)
        policy_score = discriminator.score(policy_transition, detach=True)
        expert_acc = (expert_score > 0.5).float().mean().item()
        policy_acc = (policy_score < 0.5).float().mean().item()
    return expert_acc, policy_acc


# ---------------------------------------------------------------------------
# Style reward
# ---------------------------------------------------------------------------

def amp_style_reward(
    discriminator: AMPDiscriminator,
    current_state: torch.Tensor | Sequence[torch.Tensor],
    next_state: torch.Tensor | None = None,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute AMP style reward from discriminator score.

    r = exp(-temperature * (score - 1)^2)

    Args:
        discriminator: the AMP discriminator.
        current_state: current AMP state, or transition if next_state is None.
        next_state: next AMP state (optional).
        temperature: sharpness of the reward curve (higher = stricter).
    """
    if next_state is None:
        transition = current_state
    else:
        transition = build_amp_transition(current_state, next_state)
    with torch.no_grad():
        style_score = discriminator.score(transition, detach=False)
    return torch.exp(-temperature * torch.square(style_score - 1.0))


def multi_scale_amp_style_reward(
    discriminator: MultiScaleAMPDiscriminator,
    current_state: torch.Tensor,
    next_state: torch.Tensor,
    multi_next_state: torch.Tensor | None = None,
    temperature: float = 2.0,
    scale_weights: Sequence[float] | None = None,
    return_features: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Multi-scale style reward: averages rewards from all sub-discriminators.

    Args:
        discriminator: MultiScaleAMPDiscriminator instance.
        current_state: current AMP state (batch, state_dim).
        next_state: next AMP state (1-step transition).
        multi_next_state: optional far-future state (k-step transition) for coarser scale.
        temperature: reward temperature.
        scale_weights: weights for each scale. Defaults to uniform.
        return_features: if True, return (reward, scale0_features) tuple.
    """
    if scale_weights is None:
        scale_weights = [1.0 / discriminator.num_scales] * discriminator.num_scales

    rewards = []
    s0_features = None
    # scale 0: 1-step transition
    t0 = build_amp_transition(current_state, next_state)
    with torch.no_grad():
        if return_features:
            s0_score, s0_features = discriminator.discriminators[0].score_with_features(t0)
        else:
            s0_score = discriminator.discriminators[0].score(t0, detach=False)
    r0 = torch.exp(-temperature * torch.square(s0_score - 1.0))
    rewards.append(r0 * scale_weights[0])

    # scale 1: multi-step transition (if available)
    if discriminator.num_scales > 1 and multi_next_state is not None:
        t1 = build_amp_transition(current_state, multi_next_state)
        r1 = amp_style_reward(discriminator.discriminators[1], t1, temperature=temperature)
        rewards.append(r1 * scale_weights[1])

    total = sum(rewards)
    return (total, s0_features) if return_features else total


# ---------------------------------------------------------------------------
# Enriched style reward term (replaces amp_style_reward_term)
# ---------------------------------------------------------------------------

def amp_style_reward_term(env, asset_cfg=None):
    """Reward term callable for integration with IsaacLab RewardTermCfg.

    Builds the enriched AMP state, checks 29-DoF dimensionality, computes
    style reward via attached discriminator, and maintains a FIFO buffer of
    recent policy transitions for discriminator training.

    Supports: basic state, enriched state, and multi-scale buffer.
    """
    try:
        from isaaclab.managers import SceneEntityCfg
    except Exception:
        SceneEntityCfg = None

    asset = env.scene[asset_cfg.name] if asset_cfg is not None else env.scene["robot"]

    root_pos = asset.data.root_pos_w[:, :3]
    base_height = root_pos[:, 2:3]
    dof_pos = asset.data.joint_pos
    dof_vel = asset.data.joint_vel

    num_joints = dof_pos.shape[1]
    if num_joints != 29:
        raise RuntimeError(f"AMP expects 29 joint DOF for G1, found {num_joints}.")

    # --- determine whether to use enriched state ---
    use_enriched = getattr(env, "amp_use_enriched_state", False)

    if use_enriched:
        root_quat = asset.data.root_quat_w[:, :4]
        root_lin_vel = asset.data.root_lin_vel_w[:, :3]
        root_ang_vel = asset.data.root_ang_vel_w[:, :3]
        body_pos_w = asset.data.body_pos_w

        foot_ids = getattr(env, "amp_foot_body_ids", None)
        hand_ids = getattr(env, "amp_hand_body_ids", None)

        current_state = build_amp_state(
            root_pos, dof_pos, dof_vel,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            body_pos_w=body_pos_w,
            root_pos_w=root_pos,
            foot_body_ids=foot_ids,
            hand_body_ids=hand_ids,
        )
    else:
        current_state = torch.cat((base_height, dof_pos, dof_vel), dim=-1)

    # --- initialize buffers ---
    if not hasattr(env, "amp_prev_state") or env.amp_prev_state is None:
        env.amp_prev_state = current_state.detach().cpu()
        env.amp_recent_transitions = []
        if getattr(env, "amp_use_multi_scale", False):
            env.amp_prev_states_buffer = []
            env.amp_recent_multi_transitions = []
        return torch.zeros(env.num_envs, device=env.device)

    prev_state = env.amp_prev_state.to(current_state.device)
    transition = build_amp_transition(prev_state, current_state)

    # --- push to recent transitions buffer ---
    try:
        cpu_transition = transition.detach().cpu()
        if not hasattr(env, "amp_recent_transitions") or env.amp_recent_transitions is None:
            env.amp_recent_transitions = []
        env.amp_recent_transitions.append(cpu_transition)

        num_envs_current = getattr(env, "num_envs", 1)
        max_buffer_len = max(1, 256000 // num_envs_current)
        if len(env.amp_recent_transitions) > max_buffer_len:
            env.amp_recent_transitions.pop(0)
    except Exception:
        pass

    # --- multi-scale buffer ---
    multi_prev_state = None
    if getattr(env, "amp_use_multi_scale", False):
        if not hasattr(env, "amp_prev_states_buffer") or env.amp_prev_states_buffer is None:
            env.amp_prev_states_buffer = []
        env.amp_prev_states_buffer.append(prev_state.detach().cpu())
        multi_scale_step = getattr(env, "amp_multi_scale_step", 5)
        if len(env.amp_prev_states_buffer) > multi_scale_step:
            env.amp_prev_states_buffer.pop(0)
        if len(env.amp_prev_states_buffer) >= multi_scale_step:
            multi_prev_state = env.amp_prev_states_buffer[0].to(current_state.device)
            multi_transition = build_amp_transition(multi_prev_state, current_state)
            try:
                cpu_multi = multi_transition.detach().cpu()
                if not hasattr(env, "amp_recent_multi_transitions") or env.amp_recent_multi_transitions is None:
                    env.amp_recent_multi_transitions = []
                env.amp_recent_multi_transitions.append(cpu_multi)
                max_multi_len = max(1, 128000 // getattr(env, "num_envs", 1))
                if len(env.amp_recent_multi_transitions) > max_multi_len:
                    env.amp_recent_multi_transitions.pop(0)
            except Exception:
                pass

    # --- compute style reward ---
    _disc_ok = hasattr(env, "amp_discriminator") and env.amp_discriminator is not None
    _has_prev = hasattr(env, "amp_prev_state") and env.amp_prev_state is not None
    if not _disc_ok or not _has_prev:
        if not _disc_ok:
            print(f"[AMP DEBUG] disc missing: hasattr={hasattr(env, 'amp_discriminator')}, "
                  f"env_id={id(env)}, env_type={type(env).__name__}", flush=True)
        if not _has_prev:
            print(f"[AMP DEBUG] no prev_state (first step)", flush=True)
        env.amp_prev_state = current_state.detach().cpu()
        return torch.zeros(env.num_envs, device=env.device)

    disc = env.amp_discriminator
    device = next(disc.parameters()).device if any(True for _ in disc.parameters()) else env.device
    transition_dev = transition.to(device)

    with torch.no_grad():
        score = None
        policy_feats = None
        temperature = getattr(env, "amp_style_reward_temperature", 2.0)

        if isinstance(disc, MultiScaleAMPDiscriminator):
            multi_transition_dev = None
            if multi_prev_state is not None:
                mt = build_amp_transition(multi_prev_state, current_state)
                multi_transition_dev = mt.to(device)
            fm_needed = (float(getattr(env, "amp_feature_matching_alpha", 0.0)) > 0.0
                         and hasattr(env, "amp_expert_feat_mean")
                         and env.amp_expert_feat_mean is not None)
            style_reward = multi_scale_amp_style_reward(
                disc,
                prev_state.to(device),
                current_state.to(device),
                multi_next_state=None if multi_transition_dev is None else current_state.to(device),
                temperature=temperature,
                return_features=fm_needed,
            )
            if fm_needed:
                style_reward, policy_feats = style_reward
        else:
            fm_alpha = float(getattr(env, "amp_feature_matching_alpha", 0.0))
            if fm_alpha > 0.0 and hasattr(env, "amp_expert_feat_mean") and env.amp_expert_feat_mean is not None:
                score, policy_feats = disc.score_with_features(transition_dev)
            else:
                score = torch.sigmoid(disc(transition_dev))
            style_reward = torch.exp(-temperature * torch.square(score - 1.0))

    scale = getattr(env, "amp_style_scale", 0.0)
    style_reward = style_reward * float(scale)

    # --- feature matching reward ---
    fm_alpha = float(getattr(env, "amp_feature_matching_alpha", 0.0))
    if fm_alpha > 0.0 and policy_feats is not None:
        feat_mean = env.amp_expert_feat_mean.to(device)
        feat_std = torch.sqrt(env.amp_expert_feat_var.to(device))
        dist = torch.norm((policy_feats - feat_mean) / feat_std, dim=-1)
        fm_reward = torch.exp(-fm_alpha * dist)
        style_reward = (style_reward + fm_reward) / 2.0

    _call_count = getattr(env, "_amp_debug_count", 0)
    env._amp_debug_count = _call_count + 1
    if _call_count < 3:
        if score is not None:
            print(f"[AMP DEBUG] step={_call_count}, mean_score={torch.mean(score).item():.4f}, "
                  f"mean_reward={torch.mean(style_reward).item():.4f}, scale={scale:.4f}", flush=True)
        else:
            print(f"[AMP DEBUG] step={_call_count}, multi_scale mode, "
                  f"mean_reward={torch.mean(style_reward).item():.4f}, scale={scale:.4f}", flush=True)

    env.amp_prev_state = current_state.detach().cpu()
    return style_reward.to(env.device).view(-1)


# ---------------------------------------------------------------------------
# AmpSampleBatch / AmpExpertBuffer
# ---------------------------------------------------------------------------

@dataclass
class AmpSampleBatch:
    current_state: torch.Tensor
    next_state: torch.Tensor

    @property
    def transition(self) -> torch.Tensor:
        return build_amp_transition(self.current_state, self.next_state)


class AmpExpertBuffer:
    """Expert motion buffer for AMP discriminator training.

    Loads reference motion from .npz, pre-computes enriched AMP states
    (projected gravity, velocities, foot/hand positions) and supports
    sampling at multiple temporal scales.
    """

    def __init__(
        self,
        motion_file: str | Path,
        motion_fps: float | None = None,
        device: str | torch.device | None = None,
        use_enriched_state: bool = False,
        foot_body_names: list[str] | None = None,
        hand_body_names: list[str] | None = None,
    ):
        self.device = _as_device(device)
        self.use_enriched_state = use_enriched_state
        self.motion_file = Path(motion_file)
        if not self.motion_file.is_file():
            raise FileNotFoundError(f"AMP reference file not found: {self.motion_file}")

        loaded = np.load(self.motion_file, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        elif hasattr(loaded, "item") and not isinstance(loaded, dict):
            loaded = loaded.item()

        self._data = loaded
        self.motion_fps = float(loaded.get("fps", motion_fps if motion_fps is not None else 0.0))
        if self.motion_fps <= 0.0:
            raise ValueError("AMP reference data must provide fps or motion_fps must be specified.")

        dof_pos_key = "dof_pos" if "dof_pos" in loaded else "joint_pos"
        dof_vel_key = "dof_vel" if "dof_vel" in loaded else "joint_vel"
        root_pos_key = "root_pos" if "root_pos" in loaded else "body_pos_w"
        root_rot_key = "root_rot" if "root_rot" in loaded else "body_quat_w"

        self.dof_pos = torch.as_tensor(loaded[dof_pos_key], dtype=torch.float32, device=self.device)
        self.dof_vel = torch.as_tensor(loaded[dof_vel_key], dtype=torch.float32, device=self.device)

        root_pos_tensor = torch.as_tensor(loaded[root_pos_key], dtype=torch.float32, device=self.device)
        root_rot_tensor = torch.as_tensor(loaded[root_rot_key], dtype=torch.float32, device=self.device)

        if root_pos_tensor.ndim == 3:
            self.body_pos_w_full = root_pos_tensor
            root_pos_tensor = root_pos_tensor[:, 0, :]
        else:
            self.body_pos_w_full = None
        if root_rot_tensor.ndim == 3:
            self.body_quat_w_full = root_rot_tensor
            root_rot_tensor = root_rot_tensor[:, 0, :]
        else:
            self.body_quat_w_full = None

        self.root_pos = root_pos_tensor
        self.root_rot = root_rot_tensor

        if self.dof_pos.ndim != 2 or self.dof_vel.ndim != 2:
            raise ValueError("dof_pos and dof_vel must have shape [frames, dof].")
        if self.dof_pos.shape != self.dof_vel.shape:
            raise ValueError("dof_pos and dof_vel must have the same shape.")
        if self.root_pos.shape[0] != self.dof_pos.shape[0]:
            raise ValueError("root_pos must have the same frame count as dof_pos.")

        self.num_frames = int(self.dof_pos.shape[0])
        if self.num_frames < 2:
            raise ValueError("AMP reference data must contain at least two frames.")

        # --- body name -> index mapping ---
        self.foot_body_ids: list[int] = []
        self.hand_body_ids: list[int] = []
        body_names = None
        if "body_names" in loaded:
            body_names = list(loaded["body_names"])
        if body_names is not None and self.body_pos_w_full is not None:
            if foot_body_names:
                for name in foot_body_names:
                    if name in body_names:
                        self.foot_body_ids.append(body_names.index(name))
            if hand_body_names:
                for name in hand_body_names:
                    if name in body_names:
                        self.hand_body_ids.append(body_names.index(name))

        # --- pre-compute enriched states ---
        if self.use_enriched_state and self.body_pos_w_full is not None and self.body_quat_w_full is not None:
            self._precompute_enriched_states()
        else:
            self._enriched_states = None
            self.state_dim = int(self.dof_pos.shape[1] * 2 + 1)
            self.transition_dim = self.state_dim * 2

    def _precompute_enriched_states(self):
        """Pre-compute enriched AMP states for all frames."""
        T = self.num_frames
        states = []
        for t in range(T):
            s = build_amp_state(
                self.root_pos[t:t+1],
                self.dof_pos[t:t+1],
                self.dof_vel[t:t+1],
                root_quat=self.root_rot[t:t+1],
                root_lin_vel=None,  # will interpolate
                root_ang_vel=None,
                body_pos_w=self.body_pos_w_full[t:t+1],
                root_pos_w=self.root_pos[t:t+1],
                foot_body_ids=self.foot_body_ids if self.foot_body_ids else None,
                hand_body_ids=self.hand_body_ids if self.hand_body_ids else None,
            )
            states.append(s)
        self._enriched_states = torch.cat(states, dim=0)
        self.state_dim = int(self._enriched_states.shape[1])
        self.transition_dim = self.state_dim * 2

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.motion_fps

    def to(self, device: str | torch.device) -> "AmpExpertBuffer":
        device = _as_device(device)
        self.device = device
        self.dof_pos = self.dof_pos.to(device)
        self.dof_vel = self.dof_vel.to(device)
        self.root_pos = self.root_pos.to(device)
        self.root_rot = self.root_rot.to(device)
        if self.body_pos_w_full is not None:
            self.body_pos_w_full = self.body_pos_w_full.to(device)
        if self.body_quat_w_full is not None:
            self.body_quat_w_full = self.body_quat_w_full.to(device)
        if self._enriched_states is not None:
            self._enriched_states = self._enriched_states.to(device)
        return self

    def _frame_at(self, frame_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx0 = torch.floor(frame_index).long().clamp(0, self.num_frames - 1)
        idx1 = (idx0 + 1).clamp(max=self.num_frames - 1)
        blend = (frame_index - idx0.to(frame_index.dtype)).unsqueeze(-1)
        root_pos = self.root_pos[idx0] * (1.0 - blend) + self.root_pos[idx1] * blend
        dof_pos = self.dof_pos[idx0] * (1.0 - blend) + self.dof_pos[idx1] * blend
        dof_vel = self.dof_vel[idx0] * (1.0 - blend) + self.dof_vel[idx1] * blend
        return root_pos, dof_pos, dof_vel

    def state_at_time(self, time_s: torch.Tensor) -> torch.Tensor:
        if self._enriched_states is not None:
            frame_index = torch.clamp(time_s, 0.0, self.duration) * self.motion_fps
            idx0 = torch.floor(frame_index).long().clamp(0, self.num_frames - 1)
            idx1 = (idx0 + 1).clamp(max=self.num_frames - 1)
            blend = (frame_index - idx0.to(frame_index.dtype)).unsqueeze(-1)
            return self._enriched_states[idx0] * (1.0 - blend) + self._enriched_states[idx1] * blend

        frame_index = torch.clamp(time_s, 0.0, self.duration) * self.motion_fps
        root_pos, dof_pos, dof_vel = self._frame_at(frame_index)
        return build_amp_state(root_pos, dof_pos, dof_vel)

    def transition_at_time(self, time_s: torch.Tensor, step_dt: float) -> AmpSampleBatch:
        current_state = self.state_at_time(time_s)
        next_state = self.state_at_time(time_s + step_dt)
        return AmpSampleBatch(current_state=current_state, next_state=next_state)

    def sample(self, batch_size: int, step_dt: float | None = None) -> AmpSampleBatch:
        if step_dt is None:
            step_dt = 1.0 / self.motion_fps
        max_start = max(self.duration - step_dt, 0.0)
        time_s = torch.rand(batch_size, device=self.device) * max_start
        return self.transition_at_time(time_s, step_dt)

    def sample_transition(self, batch_size: int, step_dt: float | None = None) -> torch.Tensor:
        return self.sample(batch_size, step_dt=step_dt).transition

    def sample_multi_scale(
        self,
        batch_size: int,
        step_dt_short: float | None = None,
        step_dt_long: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample transitions at two different time scales.

        Returns:
            (short_transition, long_transition): 1-step and k-step transitions.
        """
        if step_dt_short is None:
            step_dt_short = 1.0 / self.motion_fps
        if step_dt_long is None:
            step_dt_long = step_dt_short * 5

        max_start = max(self.duration - step_dt_long, 0.0)
        time_s = torch.rand(batch_size, device=self.device) * max_start
        short_batch = self.transition_at_time(time_s, step_dt_short)
        long_batch = self.transition_at_time(time_s, step_dt_long)
        return short_batch.transition, long_batch.transition
