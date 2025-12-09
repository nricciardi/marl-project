#!/bin/bash
#SBATCH --job-name=train_metaslot
#SBATCH --output=/homes/nricciardi/repository/tesi-magistrale/ocl/logs/train_metaslot_%j.out
#SBATCH --error=/homes/nricciardi/repository/tesi-magistrale/ocl/logs/train_metaslot_%j.err
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


python3 -O rllib_test.py