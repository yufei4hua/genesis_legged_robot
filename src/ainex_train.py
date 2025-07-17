import argparse
import os
import pickle
import shutil

from ainex_env import AinexEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 1024, # test
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 5,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "all_dof_names": [
            'head_pan', 'head_tilt',                                                  # 头部
            'l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper',       # 左手
            'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper',       # 右手
            'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ank_pitch', 'l_ank_roll',  # 左腿
            'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ank_pitch', 'r_ank_roll',  # 右腿
        ],
        "dof_names": [
            "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
            "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll"
        ],
        "foot_names": ["l_ank_roll_link", "r_ank_roll_link"],  # 根据URDF填写末端连杆名
        "knee_names": ["l_knee_link", "r_knee_link"],          # 根据URDF填写膝盖连杆名
        "penalize_contacts_on": ["base"],  
        "default_joint_angles": {
            "l_hip_yaw": 0.0, "l_hip_roll": 0.0, "l_hip_pitch": 0.0, "l_knee": 0.0, "l_ank_pitch": 0.0, "l_ank_roll": 0.0,
            "r_hip_yaw": 0.0, "r_hip_roll": 0.0, "r_hip_pitch": 0.0, "r_knee": 0.0, "r_ank_pitch": 0.0, "r_ank_roll": 0.0
        },
        "kp": 300.0,
        "kd": 20.0,
        "num_actions": 12,
        "clip_actions": 1.0,
        "action_scale": 1.0,
        "base_init_pos": [0.0, 0.0, 0.25],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 10.0,
        "resampling_time_s": 2.0,
        "termination_if_pitch_greater_than": 3.7,
        "termination_if_roll_greater_than": 3.7,
    }
    obs_cfg = {
        "num_single_obs": 61,
        "num_single_privileged_obs": 64,
        "frame_stack": 5,
        "c_frame_stack": 5,
        "num_obs": 61*5,
        "num_privileged_obs": 64*5,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "cycle_time": 1.96,
        "target_joint_pos_scale": 0.2,
        "max_contact_force": 200.,
        "tracking_sigma": 5.,
        "base_height_target": 0.83,
        "feet_height_target": 0.08,
        "soft_torque_limit": 0.9,
        "min_distance": 0.1,
        "max_distance": 0.5,
        "reward_scales": {
            "joint_pos": 1.5,
            "feet_contact_number": 1.0,
            
            "feet_air_time": 1.0,
            "foot_slip": -0.05,
            "feet_clearance": 1.0,
            "feet_distance": 0.2,
            "knee_distance": 0.2,
                        
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 1.0,
            "vel_mismatch_exp": 0.5,
            "low_speed": 0.2,
            "track_vel_hard": 0.5,
            
            "default_joint_pos": 0.5,
            "orientation": 1.0,
            "base_height": 0.2,
            
            "base_acc": 0.2,
            "feet_contact_forces": 0,#-2e-5,
            "action_smoothness": 0,#-2e-3,
            "torques": 0,#-2e-3,
            "dof_vel": 0,#-5e-4,
            "dof_acc": 0,#-1e-7,
            "collision": -1.,
            "torque_rate": 0,#-2e-5
            "torque_limits": -1., 
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }
    # command_cfg = {
    #     "num_commands": 3,
    #     "lin_vel_x_range": [0.5, 0.5],
    #     "lin_vel_y_range": [0., 0.],
    #     "ang_vel_range": [0.0, 0.0],
    # }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ainex-walking-all")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = AinexEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        show_viewer=not args.headless,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python src/ainex_train.py
"""