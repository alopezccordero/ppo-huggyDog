#!/bin/bash
#SBATCH --job-name=huggydog-ppo
#SBATCH --output=huggy-ppo.out.%j
#SBATCH --partition=gpua30q
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate huggy-rl


echo "Python"
which python
python --version

echo "go into the ml-agents folder"
cd ~/ppo-huggyDog/ml-agents

echo "Starting huggydog training"

mlagents-learn ./config/ppo/Huggy.yaml \
    --env=./trained-envs-executables/linux/Huggy/Huggy \
    --run-id=huggy-ppo-run-1 \
    --train \
    --no-graphics

echo "training finished"
conda deactivate