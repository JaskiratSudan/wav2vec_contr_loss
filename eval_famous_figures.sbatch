#!/bin/bash
#SBATCH --job-name=ff_eval
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=36G
#SBATCH --cpus-per-task=4
#SBATCH --time=07:00:00
#SBATCH --output=eval_famous_figures_log/%x-%j.log

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
EXP_NAMES=(
  # supcon_temp_0.05
  # supcon_temp_0.07
  # supcon_temp_0.1
  supcon
  # supcon_temp_0.6
  # supcon_geodesic_temp_0.05
  # supcon_geodesic_temp_0.07
  # supcon_geodesic_temp_0.1
  supcon_geodesic
  # supcon_geodesic_temp_0.6
  # supcon_uniformity_weight_0.01
  # supcon_uniformity_weight_0.05
  supcon_uniformity_weight_0.1
  supcon_uniformity
  supcon_uniformity_weight_0.6
)

MODEL=facebook/wav2vec2-xls-r-300m
RUN_TAG=${MODEL//\//__}

PROTOCOL_FILE=/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt
ROOT_DIR=""
SUBSET=all

SCORE_BASE_DIR=/home/jsudan/wav2vec_contr_loss/scores

BATCH_SIZE=32
NUM_WORKERS=4
MAX_DURATION_SECONDS=5
TARGET_SAMPLE_RATE=16000

#----------------------------------------------------------------#
#  RUN EVALUATION
#----------------------------------------------------------------#
cd ~/wav2vec_contr_loss || exit 1

for EXP_NAME in "${EXP_NAMES[@]}"; do
  echo "----------------------------------------------------------------"
  echo "EXPERIMENT NAME: ${EXP_NAME}"

  STAGE1_CKPT=/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/${EXP_NAME}/${RUN_TAG}/${RUN_TAG}_stage1_head_best.pt
  STAGE2_CKPT=/home/jsudan/wav2vec_contr_loss/checkpoints_stage2/${EXP_NAME}/${MODEL}/stage2_binary_head_best.pt

  SCORE_PATH=${SCORE_BASE_DIR}/${EXP_NAME}/famous_figures/score_cm_ff.txt

  if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "[ERROR] Stage-1 checkpoint not found: ${STAGE1_CKPT}"
    continue
  fi
  if [ ! -f "${STAGE2_CKPT}" ]; then
    echo "[ERROR] Stage-2 checkpoint not found: ${STAGE2_CKPT}"
    continue
  fi
  if [ ! -f "${PROTOCOL_FILE}" ]; then
    echo "[ERROR] Protocol file not found: ${PROTOCOL_FILE}"
    exit 1
  fi

  python eval_famous_figures_score_file.py \
    --protocol_file "${PROTOCOL_FILE}" \
    --root_dir "${ROOT_DIR}" \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --stage2_ckpt "${STAGE2_CKPT}" \
    --score_path "${SCORE_PATH}" \
    --model_name "${MODEL}" \
    --subset "${SUBSET}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_duration_seconds "${MAX_DURATION_SECONDS}" \
    --target_sample_rate "${TARGET_SAMPLE_RATE}" \
    --print_eer
done

echo "----------------------------------------------------------------"
echo "Evaluation finished at: $(date)"
