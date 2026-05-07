#!/bin/bash
#SBATCH --job-name=laq_analysis
#SBATCH --output=/cluster/scratch/udemirbas/LAPA/logs/analysis_%j.txt
#SBATCH --error=/cluster/scratch/udemirbas/LAPA/logs/analysis_%j.err
#SBATCH --time=02:30:00
#SBATCH --gpus=a100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=udemirbas@ethz.ch
#SBATCH --mail-type=ALL

# Load modules
module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1
module load python/3.10.13

# Activate venv
source ~/laq/bin/activate

# Set up environment
unset CUDNN_ROOT
unset CUDNN_HOME
export LD_LIBRARY_PATH="$(python -c 'import os,site; print(os.path.join(site.getsitepackages()[0], "nvidia/cudnn/lib"))'):${LD_LIBRARY_PATH}"

cd /cluster/scratch/udemirbas/LAPA/laq

export WANDB_MODE=online
export WANDB_PROJECT=phenaki_cnn
export WANDB_ENTITY=umutdemirbas-eth-z-rich
export WANDB_DIR=/cluster/scratch/udemirbas/LAPA/wandb

# Verify GPU is accessible
echo "GPU availability check:"
nvidia-smi

echo ""
echo "Running LAQ latent analysis by action/verb with GPU support..."
echo "========================================="

# Run the analysis
python3 analyze_latent_by_action.py \
	--checkpoint pre_model/laq_openx.pt \
	--split validation \
	--max-samples 6000 \
	--batch-size 32 \
	--num-workers 2 \
	--output-dir latent_analysis_results_openx \

echo ""
echo "Analysis complete. Results saved to latent_analysis_results/"
