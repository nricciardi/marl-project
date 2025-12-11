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
from_checkpoint=$2

echo "Resuming training from checkpoint: $from_checkpoint"

python3 -O ./src/multiwalker/train.py \
    --mode shared \
    --checkpoint-dir $checkpoint_dir \
    --from-checkpoint $from_checkpoint \
    --iters 300 \
    --save-interval 10 \
    --env-runners 6 \
    --num-envs-per-env-runner 20 \
    --num-cpus-per-env-runner 1 \
    --num-gpus-per-env-runner 0 \
    --lr 0.00005 \
    --gamma 0.99 \
    --training-batch-size 30720 \
    --epochs 5 \
    --num-learners 1 \
    --num-gpus-per-learner 0.5 \
    --num-cpus-per-learner 1 \
    --entropy-coeff 0.005 \
    --minibatch-size 2048 \
    --n-walkers 3