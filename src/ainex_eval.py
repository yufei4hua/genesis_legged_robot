import argparse
import os
import pickle

import torch
from ainex_env import AinexEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

import logging
logging.getLogger().setLevel(logging.WARNING)

from ainex_train import get_cfgs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ainex-walking-all")
    parser.add_argument("--ckpt", type=int, default=200)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env = AinexEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
src/ainex_eval.py -e ainex-walking-all --ckpt 2000
"""