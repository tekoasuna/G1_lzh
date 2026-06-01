import os
import yaml
import numpy as np
import pandas as pd
import onnxruntime
import torch
import time

from common.path_config import PROJECT_ROOT
from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand
from scipy.spatial.transform import Rotation as R, Slerp

class MotionLoader:
    def __init__(self, motion_file, fps=60.0):
        self.dt = 1.0 / fps
        df = pd.read_csv(motion_file, header=None)
        data = df.to_numpy(dtype=np.float32)
        
        self.num_frames = data.shape[0]
        self.duration = self.num_frames * self.dt
        
        self.root_positions = data[:, 0:3]
        # CSV format: x, y, z, qx, qy, qz, qw
        # scipy.spatial.transform.Rotation expects x,y,z,w by default
        self.root_quats = data[:, 3:7] 
        self.dof_positions = data[:, 7:]
        
        # finite difference for velocities
        self.dof_velocities = np.zeros_like(self.dof_positions)
        self.dof_velocities[:-1] = (self.dof_positions[1:] - self.dof_positions[:-1]) / self.dt
        self.dof_velocities[-1] = self.dof_velocities[-2]
        
        # Setup fast interpolation
        times = np.arange(self.num_frames) * self.dt
        self.slerp = Slerp(times, R.from_quat(self.root_quats))
        
    def update(self, t):
        t = np.clip(t, 0.0, self.duration - 1e-5)
        idx0 = int(t / self.dt)
        idx1 = min(idx0 + 1, self.num_frames - 1)
        blend = (t - idx0 * self.dt) / self.dt
        
        root_pos = self.root_positions[idx0] * (1 - blend) + self.root_positions[idx1] * blend
        dof_pos = self.dof_positions[idx0] * (1 - blend) + self.dof_positions[idx1] * blend
        dof_vel = self.dof_velocities[idx0] * (1 - blend) + self.dof_velocities[idx1] * blend
        
        root_quat = self.slerp([t])[0].as_quat() # x, y, z, w
        # BeyondMimic functions expect w, x, y, z
        root_quat_wxyz = np.array([root_quat[3], root_quat[0], root_quat[1], root_quat[2]], dtype=np.float32)
        
        return root_pos, root_quat_wxyz, dof_pos, dof_vel

class OurDance(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
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
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
            
            motion_path = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "unitree_rl_lab", "deploy", "robots", "g1_29dof", "config", "policy", "mimic", "dance_102", "params", config["motion_file"]))
            self.motion = MotionLoader(motion_path, fps=30.0)
            
            self.kps_lab = np.array(config["kp_lab"], dtype=np.float32)
            self.kds_lab = np.array(config["kd_lab"], dtype=np.float32)
            self.default_angles_lab = np.array(config["default_angles_lab"], dtype=np.float32)
            self.mj2lab = np.array(config["mj2lab"], dtype=np.int32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.action_scale_lab = np.array(config["action_scale_lab"], dtype=np.float32)
            
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            self.input_name = self.ort_session.get_inputs()[0].name

            self.action = np.zeros(self.num_actions, dtype=np.float32)
            
            print("OurDance Python policy initializing ...")
    
    def enter(self):
        self.motion_time = 0.0
        self.counter_step = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        
        # calculate initial world to init transform
        ref_root_pos, ref_root_quat_wxyz, ref_dof_pos, ref_dof_vel = self.motion.update(0.0)
        
        # Ref Torso Yaw computation
        # In Lab order, waist joints are 2 (yaw), 5 (roll), 8 (pitch)
        # But ref_dof_pos is in Mujoco order! So we index it using mj2lab
        ref_lab = ref_dof_pos[self.mj2lab]
        r_yaw = self.euler_single_axis_to_quat(ref_lab[2], 'z')
        r_roll = self.euler_single_axis_to_quat(ref_lab[5], 'x')
        r_pitch = self.euler_single_axis_to_quat(ref_lab[8], 'y')
        temp1 = self.quat_mul(r_roll, r_pitch)
        temp2 = self.quat_mul(r_yaw, temp1)
        ref_anchor_ori_w = self.quat_mul(ref_root_quat_wxyz, temp2)
        
        robot_quat = self.state_cmd.base_quat # wxyz
        qj = self.state_cmd.q[self.mj2lab] # Lab order
        qj = qj - self.default_angles_lab
        r_yaw_rob = self.euler_single_axis_to_quat(qj[2], 'z')
        r_roll_rob = self.euler_single_axis_to_quat(qj[5], 'x')
        r_pitch_rob = self.euler_single_axis_to_quat(qj[8], 'y')
        t1 = self.quat_mul(r_roll_rob, r_pitch_rob)
        t2 = self.quat_mul(r_yaw_rob, t1)
        robot_torso_ori_w = self.quat_mul(robot_quat, t2)
        
        init_to_anchor = self.matrix_from_quat(self.yaw_quat(ref_anchor_ori_w))
        world_to_anchor = self.matrix_from_quat(self.yaw_quat(robot_torso_ori_w))
        self.init_to_world = world_to_anchor @ init_to_anchor.T

    def run(self):
        self.motion_time = self.counter_step * 0.02
        if self.motion_time > self.motion.duration:
            # Reached end of motion, loop back or stay
            self.motion_time = self.motion_time % self.motion.duration
            
        ref_root_pos, ref_root_quat_wxyz, ref_dof_pos, ref_dof_vel = self.motion.update(self.motion_time)
        
        ref_lab_pos = ref_dof_pos[self.mj2lab]
        ref_lab_vel = ref_dof_vel[self.mj2lab]
        
        # Calculate ref anchor ori w
        r_yaw = self.euler_single_axis_to_quat(ref_lab_pos[2], 'z')
        r_roll = self.euler_single_axis_to_quat(ref_lab_pos[5], 'x')
        r_pitch = self.euler_single_axis_to_quat(ref_lab_pos[8], 'y')
        temp1 = self.quat_mul(r_roll, r_pitch)
        temp2 = self.quat_mul(r_yaw, temp1)
        ref_anchor_ori_w = self.quat_mul(ref_root_quat_wxyz, temp2)

        # Calculate robot torso ori w
        robot_quat = self.state_cmd.base_quat
        qj = self.state_cmd.q[self.mj2lab]
        qj = qj - self.default_angles_lab
        r_yaw_rob = self.euler_single_axis_to_quat(qj[2], 'z')
        r_roll_rob = self.euler_single_axis_to_quat(qj[5], 'x')
        r_pitch_rob = self.euler_single_axis_to_quat(qj[8], 'y')
        t1 = self.quat_mul(r_roll_rob, r_pitch_rob)
        t2 = self.quat_mul(r_yaw_rob, t1)
        robot_torso_ori_w = self.quat_mul(robot_quat, t2)
        
        # Calculate anchor relative feature
        motion_anchor_ori_b = self.matrix_from_quat(robot_torso_ori_w).T @ self.init_to_world @ self.matrix_from_quat(ref_anchor_ori_w)
        
        ang_vel = self.state_cmd.ang_vel
        dqj = self.state_cmd.dq[self.mj2lab]
        
        # Construct 154-dim observation
        # motion_command: 58 (29 pos + 29 vel)
        # motion_anchor_ori_b: 6 (first two columns of rotation matrix)
        # base_ang_vel: 3
        # joint_pos_rel: 29
        # joint_vel_rel: 29
        # last_action: 29
        obs_buf = np.concatenate((
            ref_lab_pos,
            ref_lab_vel,
            motion_anchor_ori_b[:,:2].reshape(-1),
            ang_vel,
            qj,
            dqj,
            self.action
        ), axis=-1, dtype=np.float32)
        
        obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
        observation = {self.input_name: obs_tensor}
        outputs_result = self.ort_session.run(None, observation)
        
        self.action = outputs_result[0].squeeze(0)
        
        target_dof_pos_mj = np.zeros(29)
        target_dof_pos_lab = self.action * self.action_scale_lab + self.default_angles_lab
        target_dof_pos_mj[self.mj2lab] = target_dof_pos_lab
        
        self.policy_output.actions = target_dof_pos_mj
        self.policy_output.kps[self.mj2lab] = self.kps_lab
        self.policy_output.kds[self.mj2lab] = self.kds_lab
        
        self.counter_step += 1

    def exit(self):
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.motion_time = 0
        self.counter_step = 0
        print("OurDance exited")
        
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.LOCO):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif(self.state_cmd.skill_cmd == FSMCommand.POS_RESET):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_OUR_DANCE
            
    # Quaternion and Matrix utilities (adapted from BeyondMimic.py)
    def quat_mul(self, q1, q2):
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
        return np.array([w, x, y, z])
        
    def matrix_from_quat(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])
        
    def yaw_quat(self, quat):
        rot = self.matrix_from_quat(quat)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        return self.euler_single_axis_to_quat(yaw, 'z', False)
        
    def euler_single_axis_to_quat(self, angle, axis, degrees=False):
        if degrees:
            angle = np.radians(angle)
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)
        if axis == 'x':
            return np.array([cos_half, sin_half, 0.0, 0.0])
        elif axis == 'y':
            return np.array([cos_half, 0.0, sin_half, 0.0])
        elif axis == 'z':
            return np.array([cos_half, 0.0, 0.0, sin_half])
        return np.array([1.0, 0.0, 0.0, 0.0])
