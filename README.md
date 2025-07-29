# humanoid robot walking with reinforcement learning
## installation
to run this repository, please first install genesis:

create a virtual env:

    conda create -n genesis python=3.10

stay in this repository, and run:

    pip install -e ".[dev]"
    pip install torch

then install rsl_rl lib:

    git checkout v1.0.2 && pip install -e .

## test commands
### walking velocity command 0.3m/s (can be any float between 0.0~0.3)
    python src/ainex_eval.py --ckpt 1900 --vel 0.3
### walking velocity command 0.2m/s
    python src/ainex_eval.py --ckpt 1900 --vel 0.2
### walking velocity command 0.1m/s
    python src/ainex_eval.py --ckpt 1900 --vel 0.1
## test older models (from 500~1900)
    python src/ainex_eval.py --ckpt 500

## resuls:
### results are already saved in results/ folder, please feel free to check the videos
