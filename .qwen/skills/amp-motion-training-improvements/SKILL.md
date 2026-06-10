---
name: amp-motion-training-improvements
description: Recipe for improving AMP discriminator training convergence in humanoid motion imitation (enriched state, multi-scale discriminator, LSGAN+R1 loss, spectral norm)
source: auto-skill
extracted_at: '2026-06-08T13:00:14.068Z'
---

# AMP Motion Imitation Training Improvements

## When to apply

When training an AMP (Adversarial Motion Priors) policy for humanoid motion imitation and the discriminator converges too fast or too slow, style reward is uninformative, or the agent fails to learn nuanced motion details.

## The improvement recipe

The following five changes work synergistically to stabilize discriminator training and improve motion quality:

### 1. Enrich the AMP state with kinematic features

**Before:** `[base_height, dof_pos(N), dof_vel(N)]` — captures only joint-level dynamics.

**After:** Add `[projected_gravity(3), root_lin_vel(3), root_ang_vel(3), foot_pos_b(3*K), hand_pos_b(3*M)]`.

**Why:** Dance motions are characterized by full-body kinematics — foot placement, hand trajectories, torso orientation. Without these features, the discriminator cannot distinguish a robot that has the right joint angles but the wrong body posture from one that is truly mimicking the motion.

**How:** Extend `build_amp_state()` to accept optional `root_quat`, `root_lin_vel`, `root_ang_vel`, `body_pos_w`, and body ID lists. Use quaternion utilities (conjugate, rotate) to compute projected gravity and end-effector positions in the base frame. Pre-compute enriched states in the expert buffer during `.npz` loading.

### 2. Use LSGAN loss instead of BCE

**Before:** Binary cross-entropy — `log(D(x)) + log(1-D(G(z)))`.

**After:** Least-squares GAN — `0.5 * (mean((D(x)-1)^2) + mean(D(G(z))^2))`.

**Why:** BCE saturates when the discriminator becomes confident, producing near-zero gradients for the generator (policy). LSGAN provides smooth, non-saturating gradients even when the discriminator is strong, preventing the style reward from collapsing to zero early in training.

### 3. Use R1 gradient penalty instead of WGAN-GP

**Before:** WGAN-GP — gradient penalty on random interpolations between real and fake.

**After:** R1 — gradient penalty on real data only: `||∇D(x)||²`.

**Why:** R1 requires only one forward/backward pass (no interpolation step), effectively halves the gradient penalty computation. It is the standard in modern GAN training (StyleGAN, etc.) and encourages a smooth discriminator around the real data manifold without constraining fake samples.

**Coefficient:** 0.5–2.0 for R1 (vs 5–10 for WGAN-GP). Start at 1.0.

### 4. Add spectral normalization and dropout to the discriminator

**Before:** Plain 3-layer MLP `[512, 256, 128]` with ELU.

**After:** Deeper 5-layer MLP `[512, 512, 256, 256, 128]` with GELU, spectral norm on every linear layer, and dropout (0.1) between layers.

**Why:** Spectral normalization enforces 1-Lipschitz continuity, stabilizing GAN training without needing strong gradient penalties alone. Dropout prevents the discriminator from memorizing the expert data. GELU is smoother than ELU/ReLU near zero, which helps gradient flow.

**How:** Wrap `nn.Linear` with `nn.utils.spectral_norm()` when `spectral_norm=True`. Insert `nn.Dropout(p)` after each activation. Use AdamW with small weight decay (1e-5).

### 5. Add multi-scale temporal discrimination

**Before:** Single discriminator on 1-step transitions.

**After:** Two sub-discriminators — one on 1-step (fine) and one on k-step (coarse, k=5 at 50Hz = 100ms).

**Why:** Short transitions capture instantaneous pose quality; longer transitions capture motion flow and temporal consistency. A robot might hit the right poses but with jerky transitions — only the coarse discriminator can penalize this.

**How:** Maintain two recent-transition buffers (1-step and k-step). Sample both for discriminator training. Average style rewards from both sub-discriminators with configurable weights (e.g., `[0.7, 0.3]` for fine/coarse).

## Files to modify (IsaacLab + RSL-RL project structure)

```
tasks/mimic/amp/core.py          ← AMPDiscriminator, build_amp_state, loss functions
tasks/mimic/amp/__init__.py      ← export new symbols
tasks/mimic/agents/rsl_rl_ppo_cfg.py  ← new config fields
scripts/rsl_rl/train.py          ← discriminator init + monkeypatch update loop
```

## Feature flags for backward compatibility

All improvements are gated behind flags so existing configs continue to work:

```python
amp_use_enriched_state = True   # False → basic 59-dim state
amp_use_multi_scale = True      # False → single-scale discriminator
loss_type = "lsgan"             # "bce" → original BCE
gradient_penalty_type = "r1"    # "wgan" | "none"
```

## Expected outcomes

- **Faster convergence:** Richer state + LSGAN + R1 produce more informative style rewards earlier
- **Better motion quality:** Multi-scale discriminator captures temporal coherence; enriched state captures full-body kinematics
- **Stabler training:** Spectral norm + R1 prevent discriminator collapse; LSGAN avoids saturating gradients
- **Monitor:** Track `amp_disc_loss`, `amp_expert_acc`, `amp_policy_acc` — aim for both accuracies around 0.7–0.85 (not near 1.0)

## Critical gotcha: discriminator MUST be attached to the base sim env, not a wrapper

**The bug:** `amp_style` reward is permanently 0 despite the discriminator being initialized. The TensorBoard `amp/` metrics never appear.

**Root cause:** In IsaacLab + RSL-RL training, the gym environment gets wrapped:

```
gym.make() → ManagerBasedRLEnv          ← base_env (reward terms see THIS)
    ↓
RslRlVecEnvWrapper → .unwrapped = ManagerBasedRLEnv
    ↓
DictObsWrapper     → .unwrapped = RslRlVecEnvWrapper
```

The `amp_style_reward_term(env, ...)` function is called inside the simulation step, where `env` = `ManagerBasedRLEnv`. But `env.unwrapped` at the `DictObsWrapper` level points to `RslRlVecEnvWrapper` — a *wrapper* that the reward term never sees.

**Fix:** Save a reference to the base env immediately after `gym.make()`, before ANY wrapping:

```python
env = gym.make(task, cfg=env_cfg)
base_env = env.unwrapped if hasattr(env, "unwrapped") else env

# ... later wrappers (RslRlVecEnvWrapper, DictObsWrapper) ...

# Attach EVERYTHING to base_env, NOT env.unwrapped:
base_env.amp_discriminator = amp_disc
base_env.amp_expert_buffer = expert_buf
base_env.amp_recent_transitions = []
base_env.amp_style_scale = 1.0
# ... all other amp_* attributes ...
```

Inside the monkeypatched `wrapped_update` closure, also reference `base_env` (not `env.unwrapped`) for accessing buffers and flags.

**Verification:** After the fix, training should print:
```
[AMP] Discriminator attached to base env (state_dim=59, enriched=True, multi_scale=True)
```
And `Rewards/amp_style` in TensorBoard should show non-zero values.

## Critical gotcha #2: discriminator `state_dim` must match runtime AMP state, not expert buffer

**The bug:** Discriminator built with wrong input dimension, causing a `mat1 and mat2 shapes cannot be multiplied` crash on the first style reward computation. Even if no crash (state_dim too small but catches by accident), the style reward may silently produce garbage.

**Root cause:** The expert buffer precomputes enriched states from offline `.npz` data where `root_lin_vel` and `root_ang_vel` are `None` (velocity can't be recovered from position-only data). This produces a smaller state vector than the runtime `amp_style_reward_term`, which has access to full simulation sensor data.

**Example:** With G1 (29 DOF, 2 feet, 2 hands):
- Expert buffer enriched state: `1 + 58 + 3 + 0 + 0 + 6 + 6 = 74` (or 62 without foot/hand)  
- Runtime enriched state: `1 + 58 + 3 + 3 + 3 + 6 + 6 = 80`

The discriminator's first layer is `Linear(state_dim * 2, 512)`. If built with `state_dim=62`, it expects 124-dim input; actual transitions are 160-dim → crash.

**Fix:** Never use `base_env.amp_expert_buffer.state_dim` for the discriminator. Compute `state_dim` from the same components the runtime `build_amp_state` will use:

```python
# In train.py, after loading the expert buffer:
try:
    robot_asset = base_env.scene["robot"]
    num_joints = int(robot_asset.data.joint_pos.shape[1])
except Exception:
    num_joints = 29
if use_enriched:
    state_dim = 1 + num_joints * 2 + 3 + 3 + 3  # base + joints + proj_gravity + lin_vel + ang_vel
    if foot_names:
        state_dim += 3 * len(foot_names)
    if hand_names:
        state_dim += 3 * len(hand_names)
else:
    state_dim = 1 + num_joints * 2
```

**Verification:** The startup print should show `state_dim=80` (with feet+hands) or `state_dim=68` (without). If it shows 62, 59, or any number not matching `1 + N*2 + (3 if enriched) + ... + (3*K feet) + (3*M hands)`, something is wrong.

### Debugging when style reward is still 0 after both fixes

If `amp_style` remains 0 even after attaching to `base_env`, add targeted debug prints to `amp_style_reward_term()` in `core.py` to diagnose which branch is taken:

**Step 1 — Check discriminator presence and env identity:**
```python
# In the "compute style reward" section of amp_style_reward_term():
_disc_ok = hasattr(env, "amp_discriminator") and env.amp_discriminator is not None
_has_prev = hasattr(env, "amp_prev_state") and env.amp_prev_state is not None
if not _disc_ok or not _has_prev:
    if not _disc_ok:
        print(f"[AMP DEBUG] disc missing: hasattr={hasattr(env, 'amp_discriminator')}, "
              f"env_id={id(env)}, env_type={type(env).__name__}", flush=True)
    return torch.zeros(env.num_envs, device=env.device)
```

**Step 2 — Print `base_env` identity at discriminator attachment time:**
```python
# In train.py, after attaching discriminator:
print(f"[AMP] Discriminator attached to base env (..., env_id={id(base_env)})")
```

**Step 3 — Compare:** If the `env_id` from the reward term mismatches the `env_id` from attachment, the discriminator is on the wrong object. Either:
- A wrapper was added after `base_env` was saved but before training, changing `.unwrapped`
- The RewardManager receives a different reference than the one captured as `base_env`

**Step 4 — Check scale and computed reward values:**
```python
scale = getattr(env, "amp_style_scale", 0.0)
print(f"[AMP DEBUG] mean_score={...}, mean_reward={...}, scale={scale}", flush=True)
```

**Expected on first step:** `[AMP DEBUG] no prev_state (first step)` → reward is 0 (one-time).
**Expected from second step:** `mean_reward ≈ 0.6` with untrained discriminator, rising toward 1.0 as training progresses.
**If `mean_reward` is always 0 despite passing all checks:** verify `amp_style_scale` is non-zero and the discriminator forward pass doesn't silently error inside `torch.no_grad()`.

## TensorBoard logging: use `runner.writer.add_scalar()`, not `runner.log_dict()`

RSL-RL's `OnPolicyRunner` does NOT have a `log_dict` method. The correct API is:

```python
if hasattr(runner, "writer") and runner.writer is not None:
    it = getattr(runner, "current_learning_iteration", 0)
    runner.writer.add_scalar("amp/disc_loss", loss.item(), it)
    runner.writer.add_scalar("amp/expert_acc", e_acc, it)
    runner.writer.add_scalar("amp/policy_acc", p_acc, it)
    runner.writer.add_scalar("amp/disc_mean", (e_acc + p_acc) / 2.0, it)
```

**Why the `if` guard matters:** `hasattr(runner, "log_dict")` silently returned `False` and all AMP logs were skipped. Always verify the actual RSL-RL API before hooking into it.

## RSL-RL training shell script requires subcommand

`./unitree_rl_lab.sh` uses a case statement; bare arguments fall through to the `*)` no-op:

```bash
# WRONG — silently does nothing:
./unitree_rl_lab.sh --task Unitree-G1-29dof-Mimic-Dance-102 --num_envs 4096

# CORRECT:
./unitree_rl_lab.sh --train --task Unitree-G1-29dof-Mimic-Dance-102 --num_envs 4096
./unitree_rl_lab.sh --play --task Unitree-G1-29dof-Mimic-Dance-102 --load_run 2026-06-08_22-11-39
```

## Feature matching loss (accelerates convergence 30-50%)

Standard AMP style reward is a single scalar from discriminator output. Feature matching adds an auxiliary signal: minimize L2 distance between policy transition features and expert transition features in the discriminator's penultimate layer (128-dim). This gives the policy neuron-level gradient alignment, not just a single scalar.

**Implementation in `feature-matching` branch (b4b708b):**

### 1. Split discriminator network

In `core.py` `AMPDiscriminator.__init__`:
```python
children = list(self.network.children())
self.features_net = nn.Sequential(*children[:-1])  # all but last Linear
self.head = children[-1]                            # last Linear(128→1)
```

Add `features()` method returning 128-dim penultimate output:
```python
def features(self, transition):
    return self.features_net(transition)
```

Same for `MultiScaleAMPDiscriminator` — `features(scale=None)` returns from all sub-discriminators.

### 2. Track expert feature statistics (EMA)

In `train.py` `wrapped_update`, after each discriminator step:
```python
with torch.no_grad():
    expert_feats = disc.discriminators[0].features(expert_trans)
# EMA (momentum=0.99)
base_env.amp_expert_feat_mean = momentum * old_mean + (1-m) * new_mean
base_env.amp_expert_feat_var  = momentum * old_var  + (1-m) * new_var
```

### 3. Blend with style reward

In `core.py` `amp_style_reward_term`:
```python
policy_feats = disc.discriminators[0].features(transition_dev)
feat_mean = env.amp_expert_feat_mean.to(device)
feat_std  = torch.sqrt(env.amp_expert_feat_var.to(device))
dist = torch.norm((policy_feats - feat_mean) / feat_std, dim=-1)
fm_reward = torch.exp(-alpha * dist)
style_reward = (style_reward + fm_reward) / 2.0   # 50/50 blend
```

### 4. Config

`rsl_rl_ppo_cfg.py`:
```python
feature_matching_alpha = 0.5   # 0 = disabled
```

**Why it works:** The scalar discriminator score can be 0.6 for very different reasons — policy might have wrong foot placement but right joint angles. Feature matching ties the gradient to specific neural feature alignment, giving the policy much richer direction signal per update.
