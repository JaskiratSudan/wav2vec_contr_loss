#!/bin/bash
#SBATCH --job-name=baseline_train    # Job name
#SBATCH --partition=spgpu                # GPU partition
#SBATCH --account=hafiz1               # Great Lakes account name
#SBATCH --gpus-per-node=1              # GPU per node
#SBATCH --mem=24G                      # CPU RAM
#SBATCH --cpus-per-task=4             # CPU cores (matches your num_workers)
#SBATCH --time=14:00:00                # time requested (hh:mm:ss)
#SBATCH --output=baseline_train_log/%x-%j.log             # (%x=job_name, %j=job_id)

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

MODEL=facebook/wav2vec2-xls-r-300m
RUN_TAG=${MODEL//\//__}
CKPT=/home/jsudan/wav2vec_contr_loss/checkpoints_baseline/bce/${RUN_TAG}/${RUN_TAG}_baseline_bce_best.pt
ASV_SCORE=/home/jsudan/wav2vec_contr_loss/scores/baseline/${MODEL}/score_cm_eval.txt
ITW_SCORE=/home/jsudan/wav2vec_contr_loss/scores/baseline/${MODEL}/score_cm_itw.txt

python baseline_train.py \
  --model_name ${MODEL} \
  --finetune_encoder 1 \
  --train_batch_size 32 \
  --dev_batch_size 32 \
  --enc_lr 1e-5 \
  --head_lr 5e-4

rm -rf ${ASV_SCORE}
rm -rf ${ITW_SCORE}

python eval_baseline_score_file.py \
  --ckpt ${CKPT} \
  --asv_score_path ${ASV_SCORE} \
  --itw_score_path ${ITW_SCORE} \
  --run_asv_eval \
  --run_itw

echo "----------------------------------------------------------------"
echo "Training script finished."
echo "Job finished at: $(date)"
