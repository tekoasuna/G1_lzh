---
name: amp-deployment-robustness
description: Recipe for making RoboMimic-style ONNX deployment pipeline robust against model/observation changes — auto-detect dimensions, validate shapes, and fail with actionable errors instead of silent garbage
source: auto-skill
extracted_at: '2026-06-08T13:30:00.000Z'
---

# AMP Policy Deployment Robustness (RoboMimic Deploy → MuJoCo / Real)

## When to apply

When deploying AMP-trained mimic/ dance policies via `RoboMimic_Deploy_fixed` (ONNX → MuJoCo PD control), and the deployment either:
- Crashes with cryptic dimension errors when switching to a newly trained model
- Runs but produces garbage motion because observation order drifted from training
- Fails because `StateAndCmd` lacks fields that policies expect at init time

## Root cause: silent misalignment between training and deployment observation spaces

The deployment code constructs an observation vector manually (`np.concatenate([...])`). If the training-side `PolicyCfg` observation order changes, or if the ONNX model expects a different dimension than the YAML config states, the deployment will NOT detect it — it will feed wrongly-shaped or wrongly-ordered data to the network, producing garbage actions with no error.

## Improvement recipe

### 1. Auto-detect model I/O dimensions from ONNX, not from YAML

**Before:** `self.num_obs = config["num_obs"]` — blindly trusts the YAML.

**After:**
```python
self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_info = self.ort_session.get_inputs()[0]
self.model_obs_dim = input_info.shape[1]
output_info = self.ort_session.get_outputs()[0]
self.model_act_dim = output_info.shape[1]
```

Then cross-validate against the YAML config and warn on mismatch:
```python
config_num_obs = int(config.get("num_obs", self.model_obs_dim))
if config_num_obs != self.model_obs_dim:
    print(f"[WARN] Config num_obs={config_num_obs} but ONNX expects {self.model_obs_dim}. "
          f"Using ONNX value.")
```

**Why:** The ONNX model is the ground truth. The YAML is documentation that can go stale. Trust the model, warn on config drift.

### 2. Runtime dimension validation in `run()`

After constructing the observation vector, check its length:
```python
if obs_concat.shape[0] != self.num_obs:
    raise RuntimeError(
        f"Observation dimension mismatch: built {obs_concat.shape[0]} dims, "
        f"but ONNX model expects {self.num_obs} dims.\n"
        f"Part dimensions: " + " ".join(f"{p.shape[0]}" for p in obs_parts)
    )
```

**Why:** If someone changes the observation construction (adds/removes a part, or a part changes size), this catches it immediately with diagnostic output showing which sub-block is wrong.

### 3. Warm up ONNX session at init

```python
for _ in range(10):
    self.ort_session.run(None, {self.input_name: obs_buffer.reshape(1, -1).astype(np.float32)})
```

**Why:** First ONNX inference call triggers JIT compilation / graph optimization. Without warm-up, the first `run()` call in the control loop may exceed the 20ms control deadline, causing a simulation stall or real-robot timing violation.

### 4. Declare all state fields explicitly in `StateAndCmd.__init__`

**Before:** `base_quat` was dynamically injected by `deploy_mujoco.py` via `state_cmd.base_quat = ...`. If any policy accessed it during `__init__`, it would raise `AttributeError`.

**After:**
```python
class StateAndCmd:
    def __init__(self, num_joints):
        ...
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w,x,y,z
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
```

**Why:** Policies that need the base pose (e.g., anchor orientation computation) can safely read `state_cmd.base_quat` at any time, even before the first simulation step. The identity quaternion is a safe default.

### 5. Extract full base state in deploy loop

```python
base_pos = d.qpos[:3]
base_lin_vel = d.qvel[:3]
state_cmd.base_pos = base_pos.copy()
state_cmd.base_lin_vel = base_lin_vel.copy()
```

**Why:** Some policies may need base linear velocity for velocity-tracking rewards during deployment validation. Having it available avoids a second source of `AttributeError`.

### 6. Wrap FSM run in try/except

```python
try:
    FSM_controller.run()
except Exception as e:
    print(f"[ERROR] FSM run failed: {e}")
    import traceback
    traceback.print_exc()
    # maintain last action — don't crash the simulation
```

**Why:** In MuJoCo deployment, a single inference error should not crash the viewer. Falling back to the last valid action keeps the robot standing while the error is diagnosed.

## Files to modify

```
RoboMimic_Deploy_fixed/
├── common/ctrlcomp.py              ← add base_quat, base_pos, base_lin_vel fields
├── policy/our_dance/our_dance.py   ← auto-detect dims, validate, warm-up, safe motion wrap
└── deploy_mujoco/deploy_mujoco.py  ← full base state extraction, try/except FSM call
```

## Observation order contract

The deployment observation construction MUST match the training `PolicyCfg` observation group order. For the standard mimic policy this is:

```
[motion_command(58), motion_anchor_ori_b(6), base_ang_vel(3),
 joint_pos_rel(29), joint_vel_rel(29), last_action(29)] = 154 dims
```

Document this in the policy class as `OBS_COMPONENTS` dict so future maintainers can see the expected structure at a glance.
