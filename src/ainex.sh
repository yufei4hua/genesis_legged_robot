#!/bin/bash
#SBATCH --job-name=ainex_train
#SBATCH --output=logs/ainex_train_output_%j.log
#SBATCH --error=logs/ainex_train_error_%j.log
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --qos=mcml
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate genesis

mkdir -p logs

cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97zip/genesis_legged_robot

python src/ainex_train.py