#!/bin/bash
set -e

echo "Setting up the environment..."
echo "Job started on $(hostname) at $(date)"

source ~/.myenv/bin/activate

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Current GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "----------------------------------------------------------------"

export MASTER_PORT=29500
export MASTER_ADDR=127.0.0.1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#----------------------------------------------------------------#
#  EXPERIMENT CONFIG (edit these only)
#----------------------------------------------------------------#
TRAIN_DATASET_LIST=(
  asv5
  asv19
  # mlaad
)
export TRAIN_DATASETS=$(IFS=,; echo "${TRAIN_DATASET_LIST[*]}")

export DEV_PER_DATASET=5000

EXP_NAME=asv5_bat_128_supcon_geodesic_temp_0.07
MODEL=facebook/wav2vec2-xls-r-300m

SUPCON_SIMILARITY=geodesic
TEMPERATURE=0.07

BATCH_SIZE=128
MAX_DURATION_SECONDS=10
EPOCHS=25
NUM_SAMPLES=none
SEED=1337
NUM_WORKERS=6

USE_QUEUE=0
QUEUE_SIZE=256
QUEUE_START_EPOCH=2

ACCUM_STEPS=1
CENTER_BEFORE_NORM=0
GLOBAL_CENTER=0

TOPK_NEG=1024
WARMUP_EPOCHS=100
ALPHA_END=0.9
ALPHA_RAMP_EPOCHS=20

UNIFORMITY_WEIGHT=0.0
UNIFORMITY_T=2.0

FINETUNE_ENCODER=0
ENC_LR=1e-5
HEAD_LR=5e-3
WEIGHT_DECAY=3e-3

USE_RAWBOOST=1
RAWBOOST_PROB=0.7
USE_BG_AUG=1

EVAL_DATASETS_LIST=(
  itw
  asv19
  fakexpose
  asv5
  asv21_la
  asv21_df
  famous_figures
  # mlaad
)
EVAL_DATASETS=$(IFS=, ; echo "${EVAL_DATASETS_LIST[*]}")

EVAL_BATCH_SIZE=16
EVAL_NUM_WORKERS=6

# Number of GPUs to use (adjust to match available GPUs: nvidia-smi)
NUM_GPUS=2

#----------------------------------------------------------------#
#  DERIVED / OUTPUT PATHS
#----------------------------------------------------------------#
MODEL_ID=$(basename "${MODEL}")
CKPT_ROOT=/home/jsudan/wav2vec_contr_loss/checkpoints/stage_1/${EXP_NAME}
CKPT=${CKPT_ROOT}/${MODEL_ID}_stage1_head_best.pt

ASV5_EMB_DIR=/home/jsudan/wav2vec_contr_loss/encoder_embeddings/stage1_embeddings/ASV5/${EXP_NAME}
ASV19_EMB_DIR=/home/jsudan/wav2vec_contr_loss/encoder_embeddings/stage1_embeddings/ASV19/${EXP_NAME}
ITW_EMB_DIR=/home/jsudan/wav2vec_contr_loss/encoder_embeddings/stage1_embeddings/ITW/${EXP_NAME}
MLAAD_EMB_DIR=/home/jsudan/wav2vec_contr_loss/encoder_embeddings/stage1_embeddings/MLAAD/${EXP_NAME}
POOLED_EMB_DIR=/home/jsudan/wav2vec_contr_loss/encoder_embeddings/stage1_embeddings/POOLED/${EXP_NAME}

STAGE2_DIR=checkpoints_stage2/${EXP_NAME}/wav2vec2-xls-r-300m
STAGE2_CKPT=${STAGE2_DIR}/stage2_binary_head_best.pt

#----------------------------------------------------------------#
#  DATA PATHS
#----------------------------------------------------------------#
export ASV19_TRAIN_ROOT=/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac
export ASV19_TRAIN_PROTOCOL=/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt
export ASV19_DEV_ROOT=/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac
export ASV19_DEV_PROTOCOL=/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt

export ASV5_TRAIN_ROOT=/data/Data/ASVSpoof5/No_Laundering_train/flac
export ASV5_TRAIN_PROTOCOL=/data/Data/ASVSpoof5/protocols/ASVspoof5_train_protocol.txt
export ASV5_DEV_ROOT=/data/Data/ASVSpoof5/No_Laundering_dev/flac
export ASV5_DEV_PROTOCOL=/data/Data/ASVSpoof5/protocols/ASVspoof5.dev.metadata.txt
export ASV5_EVAL_ROOT=/data/Data/ASVSpoof5/No_Laundering_eval/flac
export ASV5_EVAL_PROTOCOL=/data/Data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv

export ASV_DATASET=asv5
export ASV19_DATASET=asv19

echo "ASV5_TRAIN_ROOT=${ASV5_TRAIN_ROOT}"
echo "ASV5_TRAIN_PROTOCOL=${ASV5_TRAIN_PROTOCOL}"
echo "ASV5_DEV_ROOT=${ASV5_DEV_ROOT}"
echo "ASV5_DEV_PROTOCOL=${ASV5_DEV_PROTOCOL}"
echo "ASV5_EVAL_ROOT=${ASV5_EVAL_ROOT}"
echo "ASV5_EVAL_PROTOCOL=${ASV5_EVAL_PROTOCOL}"

#----------------------------------------------------------------#
#  RUN TRAINING
#----------------------------------------------------------------#
echo "EXPERIMENT NAME: ${EXP_NAME}"

mkdir -p train_stage1_log

cd ~/wav2vec_contr_loss

LAUNCHER="torchrun --standalone --nproc_per_node=${NUM_GPUS}"
echo "Using BATCH_SIZE=${BATCH_SIZE}, NUM_GPUS=${NUM_GPUS}"
export USE_QUEUE=${USE_QUEUE}
export QUEUE_SIZE=${QUEUE_SIZE}
export QUEUE_START_EPOCH=${QUEUE_START_EPOCH}
export CENTER_BEFORE_NORM=${CENTER_BEFORE_NORM}
export GLOBAL_CENTER=${GLOBAL_CENTER}
${LAUNCHER} train_stage1_asv5.py \
  --model_name ${MODEL} \
  --finetune_encoder ${FINETUNE_ENCODER} \
  --max_duration_seconds ${MAX_DURATION_SECONDS} \
  --save_dir ${CKPT_ROOT} \
  --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} \
  --supcon_similarity ${SUPCON_SIMILARITY} --uniformity_weight ${UNIFORMITY_WEIGHT} --uniformity_t ${UNIFORMITY_T} \
  --enc_lr ${ENC_LR} --head_lr ${HEAD_LR} --weight_decay ${WEIGHT_DECAY} --temperature ${TEMPERATURE} \
  --num_workers ${NUM_WORKERS} --seed ${SEED} \
  --topk_neg ${TOPK_NEG} --warmup_epochs ${WARMUP_EPOCHS} --alpha_end ${ALPHA_END} --alpha_ramp_epochs ${ALPHA_RAMP_EPOCHS} \
  --use_rawboost ${USE_RAWBOOST} --rawboost_prob ${RAWBOOST_PROB} \
  --use_bg_aug ${USE_BG_AUG}

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${CKPT}"
  exit 1
fi

export SKIP_ITW=0

rm -rf ${ASV5_EMB_DIR} ${ASV19_EMB_DIR} ${MLAAD_EMB_DIR} ${POOLED_EMB_DIR} ${ITW_EMB_DIR}

python extract_stage1_embeddings_generic.py \
  --dataset asv5 \
  --model_name "${MODEL}" \
  --stage1_ckpt "${CKPT}" \
  --out_dir "${ASV5_EMB_DIR}" \
  --splits "train,dev" \
  --train_root "${ASV5_TRAIN_ROOT}" --train_protocol "${ASV5_TRAIN_PROTOCOL}" \
  --dev_root   "${ASV5_DEV_ROOT}"   --dev_protocol   "${ASV5_DEV_PROTOCOL}" \
  --max_duration_seconds "${MAX_DURATION_SECONDS}"

python extract_stage1_embeddings_generic.py \
  --dataset asv19 \
  --model_name "${MODEL}" \
  --stage1_ckpt "${CKPT}" \
  --out_dir "${ASV19_EMB_DIR}" \
  --splits "train,dev" \
  --train_root "${ASV19_TRAIN_ROOT}" --train_protocol "${ASV19_TRAIN_PROTOCOL}" \
  --dev_root   "${ASV19_DEV_ROOT}"   --dev_protocol   "${ASV19_DEV_PROTOCOL}" \
  --max_duration_seconds "${MAX_DURATION_SECONDS}"

python pool_stage1_embeddings.py \
  --in_dirs "${ASV5_EMB_DIR},${ASV19_EMB_DIR}" \
  --names "asv5,asv19" \
  --out_dir "${POOLED_EMB_DIR}" \
  --splits "train,dev" \
  --allow_missing_splits

python train_stage2_classifier.py \
  --emb_dir "${POOLED_EMB_DIR}" \
  --save_dir "${STAGE2_DIR}" \
  --batch_size 128 --epochs 200 --lr 1e-4 --weight_decay 1e-4 \
  --head_type linear --hidden_dim 128 --dropout 0.2 --patience 15

if [ ! -f "${STAGE2_CKPT}" ]; then
  echo "[ERROR] Stage-2 checkpoint not found: ${STAGE2_CKPT}"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python eval_datasets.py \
  --exp_name ${EXP_NAME} \
  --datasets "${EVAL_DATASETS}" \
  --model_name ${MODEL} \
  --stage1_ckpt ${CKPT} \
  --stage2_ckpt ${STAGE2_CKPT} \
  --scores_dir /home/jsudan/wav2vec_contr_loss/scores \
  --batch_size ${EVAL_BATCH_SIZE} \
  --num_workers ${EVAL_NUM_WORKERS} \
  --max_duration_seconds ${MAX_DURATION_SECONDS} \
  --target_sample_rate 16000 \
  --print_eer \
  --asv5_root ${ASV5_EVAL_ROOT} \
  --asv5_protocol ${ASV5_EVAL_PROTOCOL} \
  --itw_root /data/Data/ds_wild/release_in_the_wild \
  --itw_protocol /data/Data/ds_wild/protocols/meta.csv

echo "----------------------------------------------------------------"
echo "Training script finished."
echo "Job finished at: $(date)"
