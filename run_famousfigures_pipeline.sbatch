#!/bin/bash
# =============================================================================
# run_famousfigures_pipeline.sh
#
# Full pipeline for FamousFigures single-speaker fine-tuning.
#
#   Step 0a — Split FamousFigures protocol into train / dev / eval
#   Step 0b — Filter ITW protocol to target speaker
#   Step 0c — Extract & cache XLSR features (sharded .pt files)
#   Step 1  — Train projection head (Stage 1) on cached XLSR features
#   Step 2  — Extract & cache projection head embeddings (256-dim)
#   Step 3  — Train classifier head (Stage 2) on cached embeddings
#   Step 4  — Final evaluation on held-out FamousFigures eval set
#   Step 5  — Evaluate fine-tuned model on ITW (speaker-filtered)
#
# Submit:
#   sbatch run_famousfigures_pipeline.sh
# =============================================================================

#SBATCH --job-name=speaker_specific_pipeline
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=train_stage1_log/%x-%j.log
#SBATCH --mail-user=jsudan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Setting up the environment..."
echo "Job started on $(hostname) at $(date)"

module purge
module load python/3.9.12
module load cuda/11.8.0
module load ffmpeg
source ~/myenv/bin/activate

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Current GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "----------------------------------------------------------------"

# =============================================================================
# SECTION 1 — Project directory
# =============================================================================
PROJECT_DIR="/home/jsudan/wav2vec_contr_loss"

# =============================================================================
# SECTION 2 — Speaker & protocol paths
# =============================================================================
SPEAKER="Donald_Trump"

# Speaker name as it appears in ITW meta.csv (spaces, not underscores)
ITW_SPEAKER_NAME="Donald Trump"

FF_PROTOCOL="/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt"
FF_ROOT="/nfs/turbo/umd-hafiz/issf_server_data/famousfigures"

ITW_PROTOCOL="/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"
ITW_ROOT="/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"

SPLITS_DIR="${PROJECT_DIR}/famous_figures/splits"
FF_PROTOCOL_TRAIN="${SPLITS_DIR}/${SPEAKER}_train.txt"
FF_PROTOCOL_DEV="${SPLITS_DIR}/${SPEAKER}_dev.txt"
FF_PROTOCOL_EVAL="${SPLITS_DIR}/${SPEAKER}_eval.txt"
ITW_PROTOCOL_SPEAKER="${SPLITS_DIR}/${SPEAKER}_itw.txt"

# FamousFigures split ratios
TRAIN_SPLIT=0.70
DEV_SPLIT=0.10
# eval gets the remaining 20%

# Which FamousFigures splits to (re)generate
# Set to "--splits train dev eval" for all three
# Set to "--splits eval" to only regenerate eval (e.g. if train/dev already exist)
SPLIT_ARGS="--splits train dev eval"

# =============================================================================
# SECTION 3 — Stage 1 checkpoint to resume from (frozen XLSR weights)
# =============================================================================
RESUME_CKPT="/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1/asv5_hbm_1024_mem_16384_supcon_geodesic_temp_0.07/wav2vec2-xls-r-300m_stage1_head_best.pt"

# =============================================================================
# SECTION 4 — Cache paths
# =============================================================================
CACHE_DIR="/nfs/turbo/umd-hafiz/jsudan/encoder_embeddings/speaker_specific_finetune_embeddings/famousfigures/cache_${SPEAKER}"

XLSR_CACHE_TRAIN="${CACHE_DIR}/xlsr_train.pt"
XLSR_CACHE_DEV="${CACHE_DIR}/xlsr_dev.pt"
XLSR_CACHE_EVAL="${CACHE_DIR}/xlsr_eval.pt"

HEAD_EMB_CACHE_TRAIN="${CACHE_DIR}/head_emb_train.pt"
HEAD_EMB_CACHE_DEV="${CACHE_DIR}/head_emb_dev.pt"
HEAD_EMB_CACHE_EVAL="${CACHE_DIR}/head_emb_eval.pt"

# =============================================================================
# SECTION 5 — XLSR model config
# =============================================================================
INPUT_DIM=1024
TARGET_SAMPLE_RATE=16000
MAX_DURATION_SECONDS=10.0

# =============================================================================
# SECTION 6 — Fake sampling (1.0 = no sampling, use all rows from protocol)
# =============================================================================
FF_FAKE_FRAC=1.0
FF_SEED=42

# =============================================================================
# SECTION 7 — Stage 1 head training hyperparameters
# =============================================================================
STAGE1_BATCH_SIZE=256
STAGE1_EPOCHS=200
STAGE1_PATIENCE=10
STAGE1_HEAD_LR=5e-4
STAGE1_WEIGHT_DECAY=3e-3
STAGE1_TEMPERATURE=0.07
STAGE1_SIMILARITY="geodesic"
STAGE1_HIDDEN_DIM=256
STAGE1_DROPOUT=0.0
STAGE1_UNIFORMITY_WEIGHT=0.0
STAGE1_UNIFORMITY_T=2.0

# =============================================================================
# SECTION 8 — Stage 2 classifier training hyperparameters
# =============================================================================
STAGE2_BATCH_SIZE=256
STAGE2_EPOCHS=100
STAGE2_PATIENCE=5
STAGE2_LR=5e-4
STAGE2_WEIGHT_DECAY=3e-3

# =============================================================================
# SECTION 9 — Output paths
# =============================================================================
RUN_ID="ff_${SPEAKER}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_ff/${RUN_ID}"

STAGE1_BEST_CKPT="${SAVE_DIR}/${RUN_ID}_stage1_head_best.pt"
STAGE2_BEST_CKPT="${SAVE_DIR}/${RUN_ID}_stage2_classifier_best.pt"
EVAL_RESULTS="${SAVE_DIR}/${RUN_ID}_eval_results.txt"
ITW_EVAL_RESULTS="${SAVE_DIR}/${RUN_ID}_itw_eval_results.txt"

# =============================================================================
# Boilerplate
# =============================================================================
mkdir -p train_stage1_log
mkdir -p "${SAVE_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${SPLITS_DIR}"

cd "${PROJECT_DIR}" || { echo "ERROR: PROJECT_DIR not found: ${PROJECT_DIR}"; exit 1; }

echo "============================================================"
echo "  Job:           ${SLURM_JOB_ID}"
echo "  Run ID:        ${RUN_ID}"
echo "  Speaker:       ${SPEAKER}"
echo "  ITW speaker:   ${ITW_SPEAKER_NAME}"
echo "  Node:          $(hostname)"
echo "  GPUs:          ${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE}}"
echo "  Save dir:      ${SAVE_DIR}"
echo "  Stage1 ckpt:   ${STAGE1_BEST_CKPT}"
echo "  Stage2 ckpt:   ${STAGE2_BEST_CKPT}"
echo "  FF Eval:       ${EVAL_RESULTS}"
echo "  ITW Eval:      ${ITW_EVAL_RESULTS}"
echo "  FF Train:      ${FF_PROTOCOL_TRAIN}"
echo "  FF Dev:        ${FF_PROTOCOL_DEV}"
echo "  FF Eval proto: ${FF_PROTOCOL_EVAL}"
echo "  ITW filtered:  ${ITW_PROTOCOL_SPEAKER}"
echo "============================================================"

# # =============================================================================
# # STEP 0a — Split FamousFigures protocol into train / dev / eval
# # =============================================================================
# echo ""
# echo "[STEP 0a] Splitting FamousFigures protocol for speaker: ${SPEAKER}..."

# python split_famousfigures.py \
#     --protocol   "${FF_PROTOCOL}" \
#     --speaker    "${SPEAKER}" \
#     --out_dir    "${SPLITS_DIR}" \
#     --train_frac "${TRAIN_SPLIT}" \
#     --dev_frac   "${DEV_SPLIT}" \
#     --seed       "${FF_SEED}" \
#     ${SPLIT_ARGS}
# [ $? -ne 0 ] && echo "ERROR: Protocol split failed." && exit 1

# echo "[STEP 0a] Done."

# # =============================================================================
# # STEP 0b — Filter ITW protocol to target speaker
# # =============================================================================
# echo ""
# echo "[STEP 0b] Filtering ITW protocol for speaker: ${ITW_SPEAKER_NAME}..."

# python filter_itw_speaker.py \
#     --protocol "${ITW_PROTOCOL}" \
#     --speaker  "${ITW_SPEAKER_NAME}" \
#     --out_dir  "${SPLITS_DIR}"
# [ $? -ne 0 ] && echo "ERROR: ITW filter failed." && exit 1

# echo "[STEP 0b] Done. ITW protocol: ${ITW_PROTOCOL_SPEAKER}"

# # =============================================================================
# # STEP 0c — Extract & cache XLSR features (all three splits in one run)
# # =============================================================================
# echo ""
# echo "[STEP 0c] Extracting XLSR features..."

# RESUME_CKPT="${RESUME_CKPT}" \
# CACHE_DIR="${CACHE_DIR}" \
# FF_TRAIN_PROTOCOL="${FF_PROTOCOL_TRAIN}" \
# FF_DEV_PROTOCOL="${FF_PROTOCOL_DEV}" \
# FF_EVAL_PROTOCOL="${FF_PROTOCOL_EVAL}" \
# FF_ROOT="${FF_ROOT}" \
# FF_FAKE_FRAC="${FF_FAKE_FRAC}" \
# FF_SEED="${FF_SEED}" \
# BATCH_SIZE=8 \
# NUM_WORKERS=2 \
# python extract_xlsr_features.py
# [ $? -ne 0 ] && echo "ERROR: XLSR extraction failed." && exit 1

# echo "[STEP 0c] Done. XLSR cache: ${CACHE_DIR}"

# # =============================================================================
# # STEP 1 — Train projection head (Stage 1) on cached XLSR features
# # =============================================================================
# echo ""
# echo "[STEP 1] Training projection head (Stage 1)..."

# CACHE_DIR="${CACHE_DIR}" \
# SAVE_DIR="${SAVE_DIR}" \
# SAVE_PATH="${STAGE1_BEST_CKPT}" \
# RESUME_HEAD_CKPT="${RESUME_CKPT}" \
# BATCH_SIZE="${STAGE1_BATCH_SIZE}" \
# EPOCHS="${STAGE1_EPOCHS}" \
# HEAD_LR="${STAGE1_HEAD_LR}" \
# WEIGHT_DECAY="${STAGE1_WEIGHT_DECAY}" \
# PATIENCE="${STAGE1_PATIENCE}" \
# TEMPERATURE="${STAGE1_TEMPERATURE}" \
# SIMILARITY="${STAGE1_SIMILARITY}" \
# UNIFORMITY_WEIGHT="${STAGE1_UNIFORMITY_WEIGHT}" \
# FF_SEED="${FF_SEED}" \
# python train_stage1_cached.py
# [ $? -ne 0 ] && echo "ERROR: Stage 1 training failed." && exit 1

# echo "[STEP 1] Done. Stage 1 checkpoint: ${STAGE1_BEST_CKPT}"

# # =============================================================================
# # STEP 2 — Extract projection head embeddings (256-dim) for all splits
# # =============================================================================
# echo ""
# echo "[STEP 2] Extracting projection head embeddings..."

# for SPLIT in train dev eval; do
#     XLSR_VAR="XLSR_CACHE_$(echo $SPLIT | tr '[:lower:]' '[:upper:]')"
#     OUT_VAR="HEAD_EMB_CACHE_$(echo $SPLIT | tr '[:lower:]' '[:upper:]')"
#     echo "  -> ${SPLIT}: ${!XLSR_VAR} -> ${!OUT_VAR}"
#     python extract_head_embeddings.py \
#         --xlsr_cache "${!XLSR_VAR}" \
#         --ckpt       "${STAGE1_BEST_CKPT}" \
#         --out_path   "${!OUT_VAR}" \
#         --hidden_dim "${STAGE1_HIDDEN_DIM}" \
#         --dropout    "${STAGE1_DROPOUT}"
#     [ $? -ne 0 ] && echo "ERROR: Head embedding extraction failed (${SPLIT})." && exit 1
# done

# echo "[STEP 2] Done."

# # =============================================================================
# # STEP 3 — Train classifier head (Stage 2)
# # =============================================================================
# echo ""
# echo "[STEP 3] Training classifier head (Stage 2)..."

# python train_stage2_cached.py \
#     --emb_cache_train "${HEAD_EMB_CACHE_TRAIN}" \
#     --emb_cache_dev   "${HEAD_EMB_CACHE_DEV}" \
#     --save_path       "${STAGE2_BEST_CKPT}" \
#     --hidden_dim      "${STAGE1_HIDDEN_DIM}" \
#     --batch_size      "${STAGE2_BATCH_SIZE}" \
#     --epochs          "${STAGE2_EPOCHS}" \
#     --patience        "${STAGE2_PATIENCE}" \
#     --lr              "${STAGE2_LR}" \
#     --weight_decay    "${STAGE2_WEIGHT_DECAY}" \
#     --seed            "${FF_SEED}"
# [ $? -ne 0 ] && echo "ERROR: Stage 2 training failed." && exit 1

# echo "[STEP 3] Done. Stage 2 checkpoint: ${STAGE2_BEST_CKPT}"

# # =============================================================================
# # STEP 4 — Final evaluation on held-out FamousFigures eval set
# # =============================================================================
# echo ""
# echo "[STEP 4] Running final evaluation on FamousFigures eval set..."

# python evaluate_stage2.py \
#     --emb_cache_eval "${HEAD_EMB_CACHE_EVAL}" \
#     --stage2_ckpt    "${STAGE2_BEST_CKPT}" \
#     --hidden_dim     "${STAGE1_HIDDEN_DIM}" \
#     --out_file       "${EVAL_RESULTS}" \
#     --speaker        "${SPEAKER}"
# [ $? -ne 0 ] && echo "ERROR: FamousFigures evaluation failed." && exit 1

# echo "[STEP 4] Done."
# echo ""
# cat "${EVAL_RESULTS}"

# =============================================================================
# STEP 5 — Evaluate fine-tuned model on ITW (speaker-filtered)
# =============================================================================
echo ""
echo "[STEP 5] Evaluating on ITW (speaker: ${ITW_SPEAKER_NAME})..."

python eval_datasets.py \
    --stage1_ckpt /scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_ff/ff_Donald_Trump_20260409_105749/ff_Donald_Trump_20260409_105749_stage1_head_best.pt \
    --stage2_ckpt /scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_ff/ff_Donald_Trump_20260409_105749/ff_Donald_Trump_20260409_105749_stage2_classifier_best.pt \
    --datasets itw \
    --itw_protocol /home/jsudan/wav2vec_contr_loss/famous_figures/splits/Donald_Trump_itw.txt \
    --itw_root /nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild \
    --print_eer
[ $? -ne 0 ] && echo "ERROR: ITW evaluation failed." && exit 1

echo "[STEP 5] Done."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "  Run ID        : ${RUN_ID}"
echo "  Save dir      : ${SAVE_DIR}"
echo "  Stage 1 ckpt  : ${STAGE1_BEST_CKPT}"
echo "  Stage 2 ckpt  : ${STAGE2_BEST_CKPT}"
echo "  FF eval       : ${EVAL_RESULTS}"
echo "  ITW protocol  : ${ITW_PROTOCOL_SPEAKER}"
echo "============================================================"