#!/bin/bash
#SBATCH --job-name=baseline_fakexpose_eval
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=eval_baseline_fakexpose_log/%x-%j.log

#SBATCH --mail-user=jsudan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Setting up the environment..."
echo "Job started on $(hostname) at $(date)"

module purge
module load python/3.9.12
module load cuda/11.8.0
source ~/myenv/bin/activate

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Current GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "----------------------------------------------------------------"

#----------------------------------------------------------------#
#  EXPERIMENT VARIABLES (EDIT THESE)
#----------------------------------------------------------------#
MODEL=facebook/wav2vec2-xls-r-300m
RUN_TAG=${MODEL//\//__}

FAKEXPOSE_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/fakexpose
SUBSET=all

CKPT=/home/jsudan/wav2vec_contr_loss/checkpoints_baseline/bce/${RUN_TAG}/${RUN_TAG}_baseline_bce_best.pt
SCORE_PATH=/home/jsudan/wav2vec_contr_loss/scores/baseline/${MODEL}/score_cm_fakexpose.txt

BATCH_SIZE=128
NUM_WORKERS=4
MAX_DURATION_SECONDS=5
TARGET_SAMPLE_RATE=16000

#----------------------------------------------------------------#
#  RUN EVALUATION
#----------------------------------------------------------------#
cd ~/wav2vec_contr_loss || exit 1

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] Baseline checkpoint not found: ${CKPT}"
  exit 1
fi
if [ ! -d "${FAKEXPOSE_ROOT}" ]; then
  echo "[ERROR] Fakexpose root not found: ${FAKEXPOSE_ROOT}"
  exit 1
fi

python eval_baseline_fakexpose_score_file.py \
  --ckpt "${CKPT}" \
  --root_dir "${FAKEXPOSE_ROOT}" \
  --score_path "${SCORE_PATH}" \
  --model_name "${MODEL}" \
  --subset "${SUBSET}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_duration_seconds "${MAX_DURATION_SECONDS}" \
  --target_sample_rate "${TARGET_SAMPLE_RATE}"

echo "----------------------------------------------------------------"
echo "Evaluation finished at: $(date)"
