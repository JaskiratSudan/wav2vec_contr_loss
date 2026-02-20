#!/bin/bash
#SBATCH --job-name=baseline_train    # Job name
#SBATCH --partition=spgpu                # GPU partition
#SBATCH --account=hafiz1               # Great Lakes account name
#SBATCH --gpus-per-node=1              # GPU per node
#SBATCH --mem=24G                      # CPU RAM
#SBATCH --cpus-per-task=4             # CPU cores (matches num_workers)
#SBATCH --time=20:00:00                # time requested (hh:mm:ss)
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
module load ffmpeg
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
SCORES_DIR=/home/jsudan/wav2vec_contr_loss/scores

# Dataset list for eval_baseline_datasets.py
DATASET_LIST=(
  itw
  asv19
  asv21_la
  asv21_df
  asv5
  famous_figures
  fakexpose
  mlaad
)
DATASETS=$(IFS=,; echo "${DATASET_LIST[*]}")

# FamousFigures speaker filter (line-by-line; leave empty for all)
FF_SPEAKERS_LIST=(
  Anthony_Blinken
  Barack_Obama
  Donald_Trump
  Elon_Musk
  JD_Vance
  Joe_Biden
  Kamala_Harris
  Mathew_Miller
  Tim_Walz
  Vivek_Ramaswamy
)
FF_SPEAKERS=$(IFS=,; echo "${FF_SPEAKERS_LIST[*]}")

python baseline_train.py \
  --model_name ${MODEL} \
  --finetune_encoder 1 \
  --train_batch_size 32 \
  --dev_batch_size 32 \
  --enc_lr 1e-5 \
  --head_lr 5e-4

rm -rf ${SCORES_DIR}

python eval_baseline_datasets.py \
  --datasets "${DATASETS}" \
  --model_name "${MODEL}" \
  --ckpt "${CKPT}" \
  --scores_dir "${SCORES_DIR}" \
  --batch_size 32 --num_workers 4 --subset all \
  --print_eer \
  --force_rescore
  # --ff_speakers "${FF_SPEAKERS}" \

echo "----------------------------------------------------------------"
echo "Training script finished."
echo "Job finished at: $(date)"
