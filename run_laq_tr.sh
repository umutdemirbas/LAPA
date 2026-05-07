#!/bin/bash
#SBATCH --job-name=laq_train
#SBATCH --output=logs/laq_train_%j.txt
#SBATCH --error=logs/laq_error_%j.txt
#SBATCH --gpus=a100:2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=62:00:00             # Adjust based on how many epochs you need
#SBATCH --mail-user=udemirbas@ethz.ch
#SBATCH --mail-type=ALL

# 1. Clean and load the exact modules needed for JAX/CUDA 12
module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1
module load python/3.10.13

# 2. Activate the environment where you ran 'pip install -e .'
source ~/laq/bin/activate

# Avoid mixing module-provided cuDNN with PyTorch wheel bundled cuDNN.
unset CUDNN_ROOT
unset CUDNN_HOME
export LD_LIBRARY_PATH="$(python -c 'import os,site; print(os.path.join(site.getsitepackages()[0], "nvidia/cudnn/lib"))'):${LD_LIBRARY_PATH}"

# 3. Navigate into the LAQ directory
cd /cluster/scratch/udemirbas/LAPA/laq

export WANDB_MODE=online
export WANDB_PROJECT=phenaki_cnn
export WANDB_ENTITY=umutdemirbas-eth-z-rich
export WANDB_DIR=/cluster/scratch/udemirbas/LAPA/wandb

# 4. Run the training using accelerate with 2 GPUs
accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=no --dynamo_backend=no train_sthv2.py