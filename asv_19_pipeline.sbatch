#!/bin/bash
#SBATCH --job-name=supcon_temp_0.3_seed_1337
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=2
#SBATCH --mem=36G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=asv_19_training_log/%x-%j.log

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

export MASTER_PORT=$((29500 + $SLURM_JOB_ID % 1000))

#----------------------------------------------------------------#
#  EXPERIMENT VARIABLES
#----------------------------------------------------------------#
EXP_NAME=supcon_temp_0.3_seed_1337

# XLS-R Layers
# export XLSR_FROM_LAYER=1
# export XLSR_TILL_LAYER=24

# SUPCON
SUPCON_SIMILARITY=cosine
TEMPERATURE=0.3
SEED=1337
# LABEL TYPE (binary | attack_type)
LABEL_TYPE=attack_type
echo "SEED: ${SEED}" 

# UNIFORMITY TERM
UNIFORMITY_WEIGHT=0.0
UNIFORMITY_T=2

# HARD NEGATIVE MINING 
TOPK_NEG=1024
WARMUP_EPOCHS=100
ALPHA_END=0.9
ALPHA_RAMP_EPOCHS=20

# MEMORY BANK
USE_QUEUE=0
QUEUE_SIZE=256
QUEUE_START_EPOCH=6

# PREVENT COLLAPSE OF SUPCON
CENTER_BEFORE_NORM=0
GLOBAL_CENTER=0

# MULTI GPU SETUP
ACCUM_STEPS=1
USE_RAVDESS=0
USE_COMMONVOICE=0
MAX_DURATION_SECONDS=10

# Plot information
PLOT_TYPE=tsne

MODEL_PATH=facebook/wav2vec2-xls-r-300m
MODEL=wav2vec2-xls-r-300m

CKPT_ROOT=/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1/${EXP_NAME}
CKPT=${CKPT_ROOT}/${MODEL}_stage1_head_best.pt

ASV_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV/${EXP_NAME}
ITW_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/${EXP_NAME}

STAGE2_DIR=checkpoints_stage2/${EXP_NAME}/${MODEL}
STAGE2_CKPT=${STAGE2_DIR}/stage2_binary_head_best.pt

SCORES_DIR=/home/jsudan/wav2vec_contr_loss/scores

WINDOWED_EVAL=0
WINDOW_SEC=${MAX_DURATION_SECONDS}
HOP_SEC=${MAX_DURATION_SECONDS}
AGG=mean
TOPK_FRAC=0.3
MAX_WINDOWS_PER_UTT=0
WINDOW_BATCH_SIZE=64

# Dataset list for eval_datasets.py
DATASET_LIST=(
  itw
  asv19
  asv21_la
  asv21_df
  asv5
  fakexpose
  deepfake_eval_2024
  famous_figures
  # mlaad
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

#----------------------------------------------------------------#
#  RUN STAGE-1 TRAINING
#----------------------------------------------------------------#
echo "EXPERIMENT NAME: ${EXP_NAME}"
echo "USE_QUEUE=${USE_QUEUE}"
echo "QUEUE_SIZE=${QUEUE_SIZE}"
echo "ACCUM_STEPS=${ACCUM_STEPS}"
echo "CENTER_BEFORE_NORM=${CENTER_BEFORE_NORM}"
echo "GLOBAL_CENTER=${GLOBAL_CENTER}"

cd ~/wav2vec_contr_loss || exit 1

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
  NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
else
  NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
fi
echo "Detected NUM_GPUS=${NUM_GPUS}"

LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --rdzv_endpoint=localhost:${MASTER_PORT}"
BATCH_SIZE=32
echo "Using BATCH_SIZE=${BATCH_SIZE}"

export USE_QUEUE=${USE_QUEUE}
export QUEUE_SIZE=${QUEUE_SIZE}
export QUEUE_START_EPOCH=${QUEUE_START_EPOCH}
export CENTER_BEFORE_NORM=${CENTER_BEFORE_NORM}
export GLOBAL_CENTER=${GLOBAL_CENTER}

${LAUNCHER} train_stage1.py \
  --model_name ${MODEL_PATH} \
  --finetune_encoder 1 \
  --save_dir ${CKPT_ROOT} \
  --epochs 100 --batch_size ${BATCH_SIZE} --num_samples none --max_duration_seconds ${MAX_DURATION_SECONDS} \
  --accum_steps ${ACCUM_STEPS} \
  --supcon_similarity ${SUPCON_SIMILARITY} --uniformity_weight ${UNIFORMITY_WEIGHT} --uniformity_t ${UNIFORMITY_T} \
  --enc_lr 1e-5 --head_lr 5e-4 --weight_decay 3e-3 --temperature ${TEMPERATURE} \
  --num_workers 4 --seed ${SEED} \
  --topk_neg ${TOPK_NEG} --warmup_epochs ${WARMUP_EPOCHS} --alpha_end ${ALPHA_END} --alpha_ramp_epochs ${ALPHA_RAMP_EPOCHS} \
  --use_rawboost 1 --rawboost_prob 0.7 \
  --use_ravdess ${USE_RAVDESS} --use_commonvoice ${USE_COMMONVOICE} \
  --label_type ${LABEL_TYPE}

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${CKPT}"
  exit 1
fi

#----------------------------------------------------------------#
#  PLOTS (ASV + ITW)
#----------------------------------------------------------------#
python plot_stage1_umap_asv.py \
  --model_name ${MODEL_PATH} \
  --ckpt_path ${CKPT} \
  --plots_dir /home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ASV/${EXP_NAME} \
  --exp_name ${EXP_NAME} \
  --dr_method ${PLOT_TYPE}
python plot_stage1_umap_itw.py \
  --model_name ${MODEL_PATH} \
  --ckpt_path ${CKPT} \
  --plots_dir /home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ITW/${EXP_NAME} \
  --exp_name ${EXP_NAME} \
  --dr_method ${PLOT_TYPE}

#----------------------------------------------------------------#
#  EXTRACT STAGE-1 EMBEDDINGS (ASV19 ONLY)
#----------------------------------------------------------------#
rm -rf ${ASV_EMB_DIR}
export SKIP_ITW=1
python extract_stage1_embeddings.py \
  --model_name ${MODEL_PATH} \
  --stage1_ckpt ${CKPT} \
  --asv_out_dir ${ASV_EMB_DIR} \
  --itw_out_dir ${ITW_EMB_DIR} \
  --max_duration_seconds ${MAX_DURATION_SECONDS}

#----------------------------------------------------------------#
#  STAGE-2 TRAINING
#----------------------------------------------------------------#
python train_stage2_classifier.py \
  --emb_dir ${ASV_EMB_DIR} \
  --save_dir ${STAGE2_DIR} \
  --batch_size 64 --epochs 200 --lr 1e-4 --weight_decay 1e-4 \
  --head_type linear --hidden_dim 128 --dropout 0.2 --patience 15

#----------------------------------------------------------------#
#  EVALUATION (eval_datasets.py)
#----------------------------------------------------------------#
python eval_datasets.py \
  --exp_name "${EXP_NAME}" \
  --datasets "${DATASETS}" \
  --model_name "${MODEL_PATH}" \
  --stage1_ckpt "${CKPT}" \
  --stage2_ckpt "${STAGE2_CKPT}" \
  --scores_dir "${SCORES_DIR}" \
  --batch_size 32 --num_workers 4 --subset all --max_duration_seconds ${MAX_DURATION_SECONDS} \
  --ff_speakers "${FF_SPEAKERS}" \
  --print_eer \
  --windowed_eval ${WINDOWED_EVAL} \
  --window_sec ${WINDOW_SEC} \
  --hop_sec ${HOP_SEC} \
  --agg ${AGG} \
  --topk_frac ${TOPK_FRAC} \
  --max_windows_per_utt ${MAX_WINDOWS_PER_UTT} \
  --window_batch_size ${WINDOW_BATCH_SIZE}

echo "----------------------------------------------------------------"
echo "ASV19 pipeline finished at: $(date)"
