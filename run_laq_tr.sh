#!/bin/bash
#SBATCH --job-name=laq_train
#SBATCH --output=logs/laq_train_%j.txt
#SBATCH --error=logs/laq_error_%j.txt
#SBATCH --gpus=a100:1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=24:00:00             # Adjust based on how many epochs you need

# 1. Clean and load the exact modules needed for JAX/CUDA 12
module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1
module load cudnn/8.9.7.29-12
module load python/3.10.13

# 2. Activate the environment where you ran 'pip install -e .'
source ~/laq/bin/activate

# 3. Navigate into the LAQ directory
cd /cluster/scratch/udemirbas/LAPA/laq

export WANDB_MODE=online
export WANDB_PROJECT=phenaki_cnn
export WANDB_ENTITY=umutdemirbas-eth-z-rich
export WANDB_DIR=/cluster/scratch/udemirbas/LAPA/wandb

# 4. Run the training using accelerate (forced to 1 GPU to avoid interactive prompts)
accelerate launch --num_processes=1 train_sthv2.py