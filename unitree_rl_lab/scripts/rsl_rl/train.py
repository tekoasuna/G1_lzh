# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Keep a reference to the base sim environment BEFORE wrapping.
    # AMP reward term runs inside the sim and sees this exact object, so the
    # discriminator, buffers, and flags MUST be attached here, not on wrappers.
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # NEW: rsl_rl 2.3.1 expects `obs` to be a dict if `obs_groups` is defined.
    # However, older IsaacLab RslRlVecEnvWrapper returns (policy_tensor, extras).
    # We must reconstruct the obs dictionary by extracting critic from extras["observations"]["critic"].
    
    class ObsDict(dict):
        def to(self, device):
            return ObsDict({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})

    class DictObsWrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.num_envs = getattr(env, "num_envs", 1)
            self.device = getattr(env, "device", "cpu")
            if hasattr(env, "get_privileged_observations"):
                self.get_privileged_observations = env.get_privileged_observations
        
        def _build_obs_dict(self, policy_tensor, extras):
            obs_dict = ObsDict({"policy": policy_tensor})
            if isinstance(extras, dict) and "observations" in extras and "critic" in extras["observations"]:
                obs_dict["critic"] = extras["observations"]["critic"]
            return obs_dict

        def step(self, actions):
            # step usually returns 4 or 5 elements
            res = self.env.step(actions)
            if len(res) == 4:
                obs, rew, done, extras = res
                return self._build_obs_dict(obs, extras), rew, done, extras
            elif len(res) == 5:
                obs, rew, term, trunc, extras = res
                return self._build_obs_dict(obs, extras), rew, term, trunc, extras
            return res
            
        def reset(self, *args, **kwargs):
            res = self.env.reset(*args, **kwargs)
            if isinstance(res, tuple) and len(res) == 2:
                obs, extras = res
                return self._build_obs_dict(obs, extras), extras
            return res
            
        def get_observations(self):
            # rsl_rl's OnPolicyRunner does `obs = env.get_observations()` (no unpacking!)
            # So we MUST return ONLY the dictionary, and discard extras here,
            # because returning a tuple makes `obs` a tuple, which crashes.
            res = self.env.get_observations()
            if isinstance(res, tuple) and len(res) == 2:
                obs, extras = res
                return self._build_obs_dict(obs, extras)
            return res
            
        def __getattr__(self, name):
            return getattr(self.env, name)

    env = DictObsWrapper(env)

    # create runner from rsl-rl
    agent_cfg_dict = agent_cfg.to_dict()
    if "obs_groups" not in agent_cfg_dict or agent_cfg_dict["obs_groups"] is None:
        agent_cfg_dict["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    # Attach AMP discriminator and expert buffer when amp_algorithm is present
    if hasattr(agent_cfg, "amp_algorithm"):
        try:
            from unitree_rl_lab.tasks.mimic.amp.core import (
                AmpExpertBuffer,
                AMPDiscriminator,
                MultiScaleAMPDiscriminator,
                amp_discriminator_loss,
                amp_discriminator_accuracy,
                r1_gradient_penalty,
                wgan_gradient_penalty,
            )
            import torch

            amp_cfg = agent_cfg.amp_algorithm

            # --- build expert buffer ---
            expert_file = amp_cfg.expert_motion_file
            expert_fps = getattr(amp_cfg, "expert_motion_fps", None)
            use_enriched = getattr(amp_cfg, "amp_use_enriched_state", False)
            foot_names = getattr(amp_cfg, "amp_foot_body_names", None)
            hand_names = getattr(amp_cfg, "amp_hand_body_names", None)

            base_env.amp_expert_buffer = AmpExpertBuffer(
                expert_file,
                motion_fps=expert_fps,
                device=agent_cfg.device,
                use_enriched_state=use_enriched,
                foot_body_names=list(foot_names) if foot_names else None,
                hand_body_names=list(hand_names) if hand_names else None,
            )

            # --- determine state dim ---
            # NOTE: The expert buffer may report a different state_dim because
            # offline data lacks root_lin_vel/root_ang_vel. Compute from the
            # runtime environment so discriminator input matches actual AMP state.
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

            # --- build discriminator ---
            use_multi_scale = getattr(amp_cfg, "amp_use_multi_scale", False)
            disc_hidden = list(getattr(amp_cfg, "discriminator_hidden_dims", [512, 256, 128]))
            disc_activation = getattr(amp_cfg, "discriminator_activation", "gelu")
            disc_sn = getattr(amp_cfg, "discriminator_spectral_norm", True)
            disc_dropout = getattr(amp_cfg, "discriminator_dropout", 0.1)

            if use_multi_scale:
                multi_scale_num = getattr(amp_cfg, "amp_multi_scale_num", 2)
                amp_disc = MultiScaleAMPDiscriminator(
                    state_dim=state_dim,
                    hidden_dims=disc_hidden,
                    activation=disc_activation,
                    spectral_norm=disc_sn,
                    dropout=disc_dropout,
                    num_scales=multi_scale_num,
                )
            else:
                amp_disc = AMPDiscriminator(
                    state_dim=state_dim,
                    hidden_dims=disc_hidden,
                    activation=disc_activation,
                    spectral_norm=disc_sn,
                    dropout=disc_dropout,
                )
            amp_disc.to(agent_cfg.device)

            # --- optimizer with weight decay ---
            disc_wd = float(getattr(amp_cfg, "discriminator_weight_decay", 0.0))
            amp_opt = torch.optim.Adam(
                amp_disc.parameters(),
                lr=amp_cfg.discriminator_learning_rate,
                weight_decay=disc_wd,
            )

            # --- attach to env ---
            base_env.amp_discriminator = amp_disc
            base_env.amp_discriminator_opt = amp_opt
            base_env.amp_style_scale = (getattr(amp_cfg, "style_reward_scale", 0.0) *
                                              getattr(amp_cfg, "reward_ratio", 1.0))
            base_env.amp_recent_transitions = []
            base_env.amp_style_reward_temperature = getattr(amp_cfg, "style_reward_temperature", 2.0)
            base_env.amp_use_enriched_state = use_enriched
            base_env.amp_use_multi_scale = use_multi_scale
            base_env.amp_multi_scale_step = getattr(amp_cfg, "amp_multi_scale_step", 5)
            print(f"[AMP] Discriminator attached to base env (state_dim={state_dim}, "
                  f"enriched={use_enriched}, multi_scale={use_multi_scale})")

            # --- body indexes for enriched state ---
            if use_enriched:
                try:
                    asset = base_env.scene["robot"]
                    body_names = list(asset.body_names) if hasattr(asset, "body_names") else []
                    if foot_names and body_names:
                        base_env.amp_foot_body_ids = [body_names.index(n) for n in foot_names if n in body_names]
                    if hand_names and body_names:
                        base_env.amp_hand_body_ids = [body_names.index(n) for n in hand_names if n in body_names]
                except Exception:
                    pass

            # --- monkeypatch runner.alg.update ---
            if hasattr(runner, "alg") and hasattr(runner.alg, "update"):
                orig_update = runner.alg.update

                def wrapped_update(*args, **kwargs):
                    result = orig_update(*args, **kwargs)

                    try:
                        batch_size = int(getattr(amp_cfg, "discriminator_batch_size", 1024))
                        expert_buf = base_env.amp_expert_buffer
                        step_dt = 1.0 / expert_buf.motion_fps
                        multi_scale = getattr(base_env, "amp_use_multi_scale", False)

                        # --- sample policy transitions ---
                        policy_buf = getattr(base_env, "amp_recent_transitions", [])
                        if len(policy_buf) == 0:
                            return result
                        buf_len = len(policy_buf)
                        num_envs = policy_buf[0].shape[0]
                        if buf_len * num_envs < batch_size:
                            return result

                        idx = torch.randint(0, buf_len, (batch_size,))
                        env_idx = torch.randint(0, num_envs, (batch_size,))
                        policy_trans = torch.stack(
                            [policy_buf[i][e] for i, e in zip(idx, env_idx)], dim=0
                        ).to(agent_cfg.device)

                        if multi_scale:
                            # sample multi-scale expert transitions
                            short_step = step_dt
                            long_step = step_dt * getattr(amp_cfg, "amp_multi_scale_step", 5)
                            expert_short, expert_long = expert_buf.sample_multi_scale(
                                batch_size, step_dt_short=short_step, step_dt_long=long_step
                            )

                            # sample multi-scale policy transitions
                            multi_buf = getattr(base_env, "amp_recent_multi_transitions", [])
                            policy_long = None
                            if len(multi_buf) > 0:
                                midx = torch.randint(0, len(multi_buf), (batch_size,))
                                policy_long = torch.stack(
                                    [multi_buf[i][e] for i, e in zip(midx, env_idx)], dim=0
                                ).to(agent_cfg.device)

                            disc = amp_disc
                            loss_type = getattr(amp_cfg, "loss_type", "lsgan")
                            gp_type = getattr(amp_cfg, "gradient_penalty_type", "r1")
                            gp_coef = float(getattr(amp_cfg, "gradient_penalty_coef", 1.0))

                            # scale 0: short-term
                            loss_s = amp_discriminator_loss(
                                disc.discriminators[0], expert_short, policy_trans, loss_type=loss_type
                            )
                            # scale 1: long-term (if available)
                            if disc.num_scales > 1 and policy_long is not None:
                                loss_l = amp_discriminator_loss(
                                    disc.discriminators[1], expert_long, policy_long, loss_type=loss_type
                                )
                                loss = loss_s + loss_l
                            else:
                                loss = loss_s

                            # R1 gradient penalty on short-scale expert only
                            if gp_type == "r1" and gp_coef > 0.0:
                                gp = r1_gradient_penalty(disc.discriminators[0], expert_short)
                                loss = loss + gp_coef * gp
                            elif gp_type == "wgan" and gp_coef > 0.0:
                                gp = wgan_gradient_penalty(disc.discriminators[0], expert_short, policy_trans)
                                loss = loss + gp_coef * gp

                        else:
                            expert_trans = expert_buf.sample_transition(batch_size, step_dt=step_dt).to(agent_cfg.device)

                            loss_type = getattr(amp_cfg, "loss_type", "lsgan")
                            gp_type = getattr(amp_cfg, "gradient_penalty_type", "r1")
                            gp_coef = float(getattr(amp_cfg, "gradient_penalty_coef", 1.0))

                            loss = amp_discriminator_loss(amp_disc, expert_trans, policy_trans, loss_type=loss_type)

                            if gp_type == "r1" and gp_coef > 0.0:
                                gp = r1_gradient_penalty(amp_disc, expert_trans)
                                loss = loss + gp_coef * gp
                            elif gp_type == "wgan" and gp_coef > 0.0:
                                gp = wgan_gradient_penalty(amp_disc, expert_trans, policy_trans)
                                loss = loss + gp_coef * gp

                        amp_opt.zero_grad()
                        loss.backward()
                        max_grad = float(getattr(amp_cfg, "max_grad_norm", 1.0))
                        torch.nn.utils.clip_grad_norm_(amp_disc.parameters(), max_grad)
                        amp_opt.step()

                        # log discriminator accuracy to TensorBoard
                        try:
                            test_batch = min(256, batch_size)
                            if multi_scale:
                                e_acc, p_acc = amp_discriminator_accuracy(
                                    disc.discriminators[0],
                                    expert_short[:test_batch],
                                    policy_trans[:test_batch],
                                )
                            else:
                                e_acc, p_acc = amp_discriminator_accuracy(
                                    amp_disc,
                                    expert_trans[:test_batch] if not multi_scale else expert_short[:test_batch],
                                    policy_trans[:test_batch],
                                )
                            if hasattr(runner, "writer") and runner.writer is not None:
                                it = getattr(runner, "current_learning_iteration", 0)
                                runner.writer.add_scalar("amp/disc_loss", loss.item(), it)
                                runner.writer.add_scalar("amp/expert_acc", e_acc, it)
                                runner.writer.add_scalar("amp/policy_acc", p_acc, it)
                                runner.writer.add_scalar("amp/disc_mean", (e_acc + p_acc) / 2.0, it)
                        except Exception:
                            pass
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[WARN] AMP Discriminator update failed: {e}")

                    return result

                runner.alg.update = wrapped_update
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] AMP initialization failed; continuing without AMP discriminator. Error: {e}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(base_env, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
