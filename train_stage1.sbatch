#!/bin/bash
#SBATCH --job-name=train_stage1    # Job name
#SBATCH --partition=spgpu                # GPU partition
#SBATCH --account=hafiz1               # Great Lakes account name
#SBATCH --gpus-per-node=1              # GPU per node
#SBATCH --mem=24G                      # CPU RAM
#SBATCH --cpus-per-task=4             # CPU cores (matches your num_workers)
#SBATCH --time=12:00:00                # time requested (hh:mm:ss)
#SBATCH --output=train_stage1_log/%x-%j.log             # (%x=job_name, %j=job_id)

# --- Email Notifications (Optional but Recommended) ---
#SBATCH --mail-user=jsudan@umich.edu  # Replace with your email
#SBATCH --mail-type=BEGIN,END,FAIL           # Send email on job start, end, and failure

#----------------------------------------------------------------#
#  SETUP THE ENVIRONMENT
#----------------------------------------------------------------#
echo "Setting up the environment..."
echo "Job started on $(hostname) at $(date)"

# Load necessary modules. This ensures a clean and consistent environment.
module purge
module load python/3.9.12
module load cuda/11.8.0

# Activate your Python virtual environment
source ~/myenv/bin/activate

# Print some diagnostic information to the log file for debugging
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Current GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "----------------------------------------------------------------"


#----------------------------------------------------------------#
#  RUN THE TRAINING SCRIPT
#----------------------------------------------------------------#
echo "Starting the Python training script..."

# Navigate to your project directory if the script is not in your home directory
cd ~/wav2vec_contr_loss

# Execute your main training script
# python train_stage1.py --model_name facebook/wav2vec2-large-960h
python train_stage1.py --model_name facebook/wav2vec2-xls-r-300m
python plot_stage1_umap_asv.py --model_name facebook/wav2vec2-xls-r-300m --ckpt_path /home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost --plots_dir /home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ASV/with_rawboost
python plot_stage1_umap_itw.py --model_name facebook/wav2vec2-xls-r-300m --ckpt_path /home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost --plots_dir /home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ITW/with_rawboost

echo "----------------------------------------------------------------"
echo "Training script finished."
echo "Job finished at: $(date)"
