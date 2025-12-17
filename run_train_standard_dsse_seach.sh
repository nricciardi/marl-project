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

python3 -O ./src/dsse_search/standard/train.py \
    --seed 42 \
    --mode shared \
    --checkpoint-dir $checkpoint_dir \
    --iters 200 \
    --save-interval 10 \
    --env-runners 6 \
    --num-envs-per-env-runner 10 \
    --num-cpus-per-env-runner 1 \
    --num-gpus-per-env-runner 0 \
    --lr 1e-4 \
    --gamma 0.995 \
    --clip-param 0.3 \
    --lambda 0.95 \
    --training-batch-size 30720 \
    --minibatch-size 3072 \
    --epochs 10 \
    --num-learners 1 \
    --num-gpus-per-learner 0.5 \
    --num-cpus-per-learner 1 \
    --entropy-coeff 0.01 \
    --probability-matrix-cnn-conv2d 1 16 32 64 \
    --probability-matrix-cnn-kernel-sizes 3 3 3 3 \
    --probability-matrix-cnn-strides 2 2 2 2 \
    --probability-matrix-cnn-paddings 1 1 1 1 \
    --drone-coordinates-mlp-hiddens 16 32 \
    --drone-coordinates-mlp-dropout 0.0 \
    --fusion-mlp-hiddens 128 64 \
    --fusion-mlp-dropout 0.0 \
    --grid-size 40 \
    --timestep-limit 100 \
    --person-amount 1 \
    --person-initial-position 15 15 \
    --person-speed 1.0 1.0 \
    --dispersion-inc 0.1 \
    --drone-amount 3 \
    --drone-speed 10 \
    --detection-probability 0.9