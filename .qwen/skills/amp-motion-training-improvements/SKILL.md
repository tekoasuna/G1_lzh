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

### Debugging when style reward is still 0 after the fix

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
