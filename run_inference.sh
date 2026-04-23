#!/bin/bash
#SBATCH --job-name=lapa_inference
#SBATCH --output=inference_output_%j.txt  # Saves your output to a text file
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --mem-per-cpu=64G                 # The RAM we found you needed
#SBATCH --time=02:00:00                   # 2 hours max run time

# 1. Clean the environment and load the exact modules that worked
module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1
module load cudnn/8.9.7.29-12
module load python/3.10.13

# 2. Put on your toolbelt
source ~/lapa/bin/activate

# 3. Navigate to your scratch folder (just to be safe)
cd /cluster/scratch/udemirbas/LAPA

# 4. Run the code (with the JAX memory fix, just in case)
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m latent_pretraining.inference