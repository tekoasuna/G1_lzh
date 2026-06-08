import os
import yaml
import numpy as np
import pandas as pd
import onnxruntime
import torch

from common.path_config import PROJECT_ROOT
from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand
from scipy.spatial.transform import Rotation as R, Slerp


class MotionLoader:
    """Loads reference motion from CSV and provides frame-accurate interpolation.

    CSV format: [root_pos_x, root_pos_y, root_pos_z, root_rot_x, root_rot_y, root_rot_z, root_rot_w, dof_0...dof_28]
    """

    def __init__(self, motion_file, fps=60.0):
        self.dt = 1.0 / fps
        df = pd.read_csv(motion_file, header=None)
        data = df.to_numpy(dtype=np.float32)

        self.num_frames = data.shape[0]
        self.duration = self.num_frames * self.dt

        self.root_positions = data[:, 0:3]
        self.root_quats_xyzw = data[:, 3:7]  # scipy expects x,y,z,w
        self.dof_positions = data[:, 7:]

        # finite-difference velocities
        self.dof_velocities = np.zeros_like(self.dof_positions)
        self.dof_velocities[:-1] = (self.dof_positions[1:] - self.dof_positions[:-1]) / self.dt
        self.dof_velocities[-1] = self.dof_velocities[-2]

        times = np.arange(self.num_frames) * self.dt
        self.slerp = Slerp(times, R.from_quat(self.root_quats_xyzw))

    def update(self, t):
        """Interpolate motion at time t (seconds).  Returns (root_pos, root_quat_wxyz, dof_pos, dof_vel)."""
        t = np.clip(t, 0.0, self.duration - 1e-5)
        idx0 = int(t / self.dt)
        idx1 = min(idx0 + 1, self.num_frames - 1)
        blend = (t - idx0 * self.dt) / self.dt

        root_pos = self.root_positions[idx0] * (1 - blend) + self.root_positions[idx1] * blend
        dof_pos = self.dof_positions[idx0] * (1 - blend) + self.dof_positions[idx1] * blend
        dof_vel = self.dof_velocities[idx0] * (1 - blend) + self.dof_velocities[idx1] * blend

        root_quat_xyzw = self.slerp([t])[0].as_quat()  # x, y, z, w
        root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]], dtype=np.float32)
        return root_pos, root_quat_wxyz, dof_pos, dof_vel


# ---------------------------------------------------------------------------
# Quaternion / matrix helpers
# ---------------------------------------------------------------------------

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.array([w, x, y, z], dtype=np.float32)


def _matrix_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ], dtype=np.float32)


def _yaw_quat(quat_wxyz):
    """Extract yaw-only quaternion [w, x, y, z]."""
    rot = _matrix_from_quat(quat_wxyz)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return _euler_single_axis_to_quat(yaw, 'z')


def _euler_single_axis_to_quat(angle, axis):
    half = angle / 2.0
    sin_h = np.sin(half)
    cos_h = np.cos(half)
    if axis == 'x':
        return np.array([cos_h, sin_h, 0.0, 0.0], dtype=np.float32)
    if axis == 'y':
        return np.array([cos_h, 0.0, sin_h, 0.0], dtype=np.float32)
    if axis == 'z':
        return np.array([cos_h, 0.0, 0.0, sin_h], dtype=np.float32)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _anchor_orientation_w(root_quat_wxyz, dof_pos_lab):
    """Compute anchor (torso) orientation in world frame."""
    r_yaw = _euler_single_axis_to_quat(dof_pos_lab[2], 'z')
    r_roll = _euler_single_axis_to_quat(dof_pos_lab[5], 'x')
    r_pitch = _euler_single_axis_to_quat(dof_pos_lab[8], 'y')
    torso_local_rot = _quat_mul(r_yaw, _quat_mul(r_roll, r_pitch))
    return _quat_mul(root_quat_wxyz, torso_local_rot)


# ---------------------------------------------------------------------------
# OurDance FSM state
# ---------------------------------------------------------------------------

class OurDance(FSMState):
    """Deploy AMP-trained mimic policy (OurDance / dance_102).

    Constructs the same observation vector as the training PolicyCfg:
      [motion_command(58), motion_anchor_ori_b(6), base_ang_vel(3),
       joint_pos_rel(29), joint_vel_rel(29), last_action(29)] = 154 dims.

    Auto-detects input dimension from ONNX model and validates config.
    """

    # Expected observation components (for reference and validation)
    OBS_COMPONENTS = {
        "motion_command": 58,       # 29 ref joint pos + 29 ref joint vel
        "motion_anchor_ori_b": 6,   # first 2 cols of relative rotation matrix
        "base_ang_vel": 3,
        "joint_pos_rel": 29,
        "joint_vel_rel": 29,
        "last_action": 29,
    }
    EXPECTED_OBS_DIM = sum(OBS_COMPONENTS.values())  # 154

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_OUR_DANCE
        self.name_str = "our_dance"
        self.counter_step = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "OurDance.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # --- paths ---
        self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
        if not os.path.isfile(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        motion_file_rel = config.get("motion_file", "xinjiang.csv")
        motion_path = os.path.abspath(os.path.join(
            PROJECT_ROOT, "..", "unitree_rl_lab", "deploy", "robots", "g1_29dof",
            "config", "policy", "mimic", "dance_102", "params", motion_file_rel,
        ))
        if not os.path.isfile(motion_path):
            raise FileNotFoundError(f"Motion CSV not found: {motion_path}")
        self.motion = MotionLoader(motion_path, fps=30.0)

        # --- robot parameters ---
        self.kps_lab = np.array(config["kp_lab"], dtype=np.float32)
        self.kds_lab = np.array(config["kd_lab"], dtype=np.float32)
        self.default_angles_lab = np.array(config["default_angles_lab"], dtype=np.float32)
        self.mj2lab = np.array(config["mj2lab"], dtype=np.int32)
        self.action_scale_lab = np.array(config["action_scale_lab"], dtype=np.float32)

        # --- ONNX model ---
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=['CPUExecutionProvider'],
        )
        input_info = self.ort_session.get_inputs()[0]
        self.input_name = input_info.name
        self.model_obs_dim = input_info.shape[1]  # batch, obs_dim

        output_info = self.ort_session.get_outputs()[0]
        self.model_act_dim = output_info.shape[1]  # batch, act_dim

        # --- config validation ---
        config_num_obs = int(config.get("num_obs", self.model_obs_dim))
        if config_num_obs != self.model_obs_dim:
            print(f"[WARN] Config num_obs={config_num_obs} but ONNX expects {self.model_obs_dim}. "
                  f"Using ONNX value {self.model_obs_dim}.")
        self.num_obs = self.model_obs_dim

        config_num_actions = int(config.get("num_actions", self.model_act_dim))
        if config_num_actions != self.model_act_dim:
            print(f"[WARN] Config num_actions={config_num_actions} but ONNX expects {self.model_act_dim}. "
                  f"Using ONNX value {self.model_act_dim}.")
        self.num_actions = self.model_act_dim

        # --- dimension sanity check ---
        if self.num_obs != self.EXPECTED_OBS_DIM:
            print(f"[WARN] ONNX expects obs_dim={self.num_obs}, "
                  f"but default mimic policy expects {self.EXPECTED_OBS_DIM}. "
                  f"Observation construction may need adjustment.")

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)

        # warm up ONNX session
        for _ in range(10):
            obs_tensor = self.obs_buffer.reshape(1, -1).astype(np.float32)
            self.ort_session.run(None, {self.input_name: obs_tensor})

        print(f"[OurDance] ONNX model loaded: obs_dim={self.num_obs}, act_dim={self.num_actions}")
        print(f"[OurDance] Motion: {self.motion.num_frames} frames, {self.motion.duration:.1f}s")

    # ------------------------------------------------------------------
    # FSM lifecycle
    # ------------------------------------------------------------------

    def enter(self):
        self.motion_time = 0.0
        self.counter_step = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)

        # compute initial world-to-torso anchor alignment
        ref_root_pos, ref_root_quat_wxyz, ref_dof_pos, ref_dof_vel = self.motion.update(0.0)
        ref_lab = ref_dof_pos[self.mj2lab] - self.default_angles_lab
        ref_anchor_ori_w = _anchor_orientation_w(ref_root_quat_wxyz, ref_lab)

        robot_quat = self.state_cmd.base_quat
        qj = self.state_cmd.q[self.mj2lab] - self.default_angles_lab
        robot_anchor_ori_w = _anchor_orientation_w(robot_quat, qj)

        init_to_anchor = _matrix_from_quat(_yaw_quat(ref_anchor_ori_w))
        world_to_anchor = _matrix_from_quat(_yaw_quat(robot_anchor_ori_w))
        self.init_to_world = world_to_anchor @ init_to_anchor.T

    def run(self):
        step_dt = 0.02  # 50 Hz control
        self.motion_time = self.counter_step * step_dt

        # wrap motion time if beyond duration
        if self.motion_time >= self.motion.duration:
            self.motion_time = self.motion_time % self.motion.duration

        ref_root_pos, ref_root_quat_wxyz, ref_dof_pos, ref_dof_vel = self.motion.update(self.motion_time)

        # reference joint state in Lab order
        ref_lab_pos = ref_dof_pos[self.mj2lab]
        ref_lab_vel = ref_dof_vel[self.mj2lab]

        # --- motion_anchor_ori_b ---
        ref_anchor_ori_w = _anchor_orientation_w(ref_root_quat_wxyz, ref_lab_pos - self.default_angles_lab)
        robot_quat = self.state_cmd.base_quat
        qj = self.state_cmd.q[self.mj2lab] - self.default_angles_lab
        robot_anchor_ori_w = _anchor_orientation_w(robot_quat, qj)

        motion_anchor_ori_b = (
            _matrix_from_quat(robot_anchor_ori_w).T
            @ self.init_to_world
            @ _matrix_from_quat(ref_anchor_ori_w)
        )

        # --- robot state ---
        ang_vel = self.state_cmd.ang_vel
        dqj = self.state_cmd.dq[self.mj2lab]

        # --- build observation ---
        # Observation order MUST match training PolicyCfg:
        #   motion_command, motion_anchor_ori_b, base_ang_vel,
        #   joint_pos_rel, joint_vel_rel, last_action
        obs_parts = [
            ref_lab_pos,                                 # 29: reference joint positions
            ref_lab_vel,                                 # 29: reference joint velocities
            motion_anchor_ori_b[:, :2].reshape(-1),      # 6: first 2 cols of rot mat
            ang_vel,                                     # 3: base angular velocity
            qj,                                          # 29: current joint pos (rel)
            dqj,                                         # 29: current joint vel (rel)
            self.action,                                 # 29: last action
        ]
        obs_concat = np.concatenate(obs_parts, axis=-1, dtype=np.float32)

        # --- dimension check ---
        if obs_concat.shape[0] != self.num_obs:
            raise RuntimeError(
                f"Observation dimension mismatch: built {obs_concat.shape[0]} dims, "
                f"but ONNX model expects {self.num_obs} dims.\n"
                f"Part dimensions: "
                + " ".join(f"{p.shape[0]}" for p in obs_parts)
            )

        # --- ONNX inference ---
        obs_tensor = obs_concat.reshape(1, -1)
        outputs_result = self.ort_session.run(None, {self.input_name: obs_tensor})
        self.action = outputs_result[0].squeeze(0).astype(np.float32)

        # --- action decode ---
        target_dof_pos_lab = self.action * self.action_scale_lab + self.default_angles_lab
        target_dof_pos_mj = np.zeros(29, dtype=np.float32)
        target_dof_pos_mj[self.mj2lab] = target_dof_pos_lab

        self.policy_output.actions = target_dof_pos_mj
        self.policy_output.kps[self.mj2lab] = self.kps_lab
        self.policy_output.kds[self.mj2lab] = self.kds_lab

        self.counter_step += 1

    def exit(self):
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)
        self.motion_time = 0.0
        self.counter_step = 0
        print("[OurDance] exited")

    def checkChange(self):
        cmd = self.state_cmd.skill_cmd
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        if cmd == FSMCommand.LOCO:
            return FSMStateName.SKILL_COOLDOWN
        if cmd == FSMCommand.PASSIVE:
            return FSMStateName.PASSIVE
        if cmd == FSMCommand.POS_RESET:
            return FSMStateName.FIXEDPOSE
        return FSMStateName.SKILL_OUR_DANCE
