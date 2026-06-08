# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # --- Discriminator architecture ---
    discriminator_hidden_dims = [512, 512, 256, 256, 128]
    discriminator_activation = "gelu"       # "elu" | "relu" | "gelu" | "leaky_relu"
    discriminator_spectral_norm = True
    discriminator_dropout = 0.1
    discriminator_learning_rate = 3.0e-4

    # --- Loss configuration ---
    loss_type = "lsgan"                     # "bce" | "lsgan"
    gradient_penalty_type = "r1"            # "r1" | "wgan" | "none"
    gradient_penalty_coef = 1.0             # R1: 0.5-2.0; WGAN: 5.0-10.0
    discriminator_batch_size = 1024

    # --- Expert data ---
    expert_motion_file = "/root/G1_Project/XingJiang002.npz"
    expert_motion_fps = 50.0

    # --- AMP state ---
    amp_state_dim = 59                      # basic: 59; enriched: 80 (auto-detected)
    amp_transition_dim = 118                # basic: 118; enriched: 160
    amp_use_enriched_state = True
    amp_foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    amp_hand_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    # --- Multi-scale discriminator ---
    amp_use_multi_scale = True
    amp_multi_scale_num = 2
    amp_multi_scale_step = 5                # 5-step (100ms at 50Hz) for coarse scale
    amp_multi_scale_weights = [0.7, 0.3]    # short-scale, long-scale weights

    # --- Style reward ---
    style_reward_temperature = 2.0
    style_reward_scale = 1.0
    reward_ratio = 1.0

    # --- Policy ---
    policy_loss_detach_discriminator = True

    # --- Regularization ---
    discriminator_weight_decay = 1e-5


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp_algorithm = AmpPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
