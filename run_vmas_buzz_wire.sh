#!/bin/bash
#SBATCH --job-name=dai
#SBATCH --output=./logs/train_%j.out
#SBATCH --error=./logs/train_%j.err
#SBATCH --account=tesi_nricciardi
#SBATCH --partition=all_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTX_A5000_24G"

module load cuda/12.6.3-none-none

export PYTHONPATH=./src:$PYTHONPATH

checkpoint_dir=$1

python3 -O ./src/vmas_buzz_wire/train.py \
    --seed 42 \
    --mode shared \
    --checkpoint-dir $checkpoint_dir \
    --iters 500 \
    --save-interval 10 \
    --env-runners 6 \
    --num-envs-per-env-runner 20 \
    --num-cpus-per-env-runner 1 \
    --num-gpus-per-env-runner 0 \
    --lr 1e-4 \
    --gamma 0.99 \
    --clip-param 0.3 \
    --lambda 0.95 \
    --training-batch-size 5000 \
    --epochs 10 \
    --num-learners 1 \
    --num-gpus-per-learner 0.5 \
    --num-cpus-per-learner 1 \
    --entropy-coeff 0.01 \
    --minibatch-size 500 \
    --wall-length 2 \
    --n-agents 2 \
    --agent-radius 0.03 \
    --ball-radius 0.03 \
    --stacked-frames 4 \
    --continuous-actions \
    --agent-spacing 0.5 \
    --fcnet-activation tanh \
    --fcnet-hiddens 400 300 \
    --kl-coeff 1.0