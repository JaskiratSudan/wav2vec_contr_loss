#!/bin/bash
#SBATCH --job-name=pooled_bat_64_supcon_geodesic_temp_0.07
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=36G
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
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

export MASTER_PORT=$((29500 + $SLURM_JOB_ID % 1000))

export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#----------------------------------------------------------------#
#  EXPERIMENT CONFIG (edit these only)
#----------------------------------------------------------------#
TRAIN_DATASET_LIST=(
  asv5
  asv19
  mlaad
)
export TRAIN_DATASETS=$(IFS=,; echo "${TRAIN_DATASET_LIST[*]}")

export DEV_PER_DATASET=5000   # it is using ASV 5 and ASV 19 dev set

EXP_NAME=pooled_bat_64_supcon_geodesic_temp_0.07
MODEL=facebook/wav2vec2-xls-r-300m

SUPCON_SIMILARITY=geodesic
TEMPERATURE=0.07

# Train hyperparams
BATCH_SIZE=64
MAX_DURATION_SECONDS=10
EPOCHS=10
NUM_SAMPLES=none
SEED=1337
NUM_WORKERS=6

# ----- QUEUE / MEMORY BANK -----
USE_QUEUE=0
QUEUE_SIZE=256
QUEUE_START_EPOCH=2

ACCUM_STEPS=1
CENTER_BEFORE_NORM=0
GLOBAL_CENTER=0

# ----- HARD BATCH MINING -----
TOPK_NEG=1024
WARMUP_EPOCHS=100
ALPHA_END=0.9
ALPHA_RAMP_EPOCHS=20

# ----- UNIFORMITY TERM -----
UNIFORMITY_WEIGHT=0.0
UNIFORMITY_T=2.0

# ----- SOFT SILHOUETTE TERM -----
SIL_WEIGHT=0.0

FINETUNE_ENCODER=1
ENC_LR=1e-5
HEAD_LR=5e-3
WEIGHT_DECAY=3e-3

USE_RAWBOOST=1
RAWBOOST_PROB=0.1
USE_BG_AUG=1

# ----- EVAL DATASETS as a LIST -----
EVAL_DATASETS_LIST=(
  itw
  asv19
  # fakexpose
  # asv5
  # asv21_df
  # asv21_la
  # mlaad
)
EVAL_DATASETS=$(IFS=, ; echo "${EVAL_DATASETS_LIST[*]}")

EVAL_BATCH_SIZE=32
EVAL_NUM_WORKERS=6

#----------------------------------------------------------------#
#  DERIVED / OUTPUT PATHS (do not edit unless needed)
#----------------------------------------------------------------#
RUN_TAG=${MODEL//\//__}

MODEL_ID=$(basename "${MODEL}")
CKPT_ROOT=/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1/${EXP_NAME}
CKPT=${CKPT_ROOT}/${RUN_TAG}/${MODEL_ID}_stage1_head_best.pt

ASV5_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV5/${EXP_NAME}
ASV19_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV19/${EXP_NAME}
ITW_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/${EXP_NAME}
MLAAD_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/MLAAD/${EXP_NAME}


POOLED_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/POOLED/${EXP_NAME}

STAGE2_DIR=checkpoints_stage2/${EXP_NAME}/wav2vec2-xls-r-300m
STAGE2_CKPT=${STAGE2_DIR}/stage2_binary_head_best.pt

SCORE_ASV=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/wav2vec2-xls-r-300m/score_cm_eval_asv5.txt
SCORE_ITW=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/wav2vec2-xls-r-300m/score_cm_itw.txt
SCORE_ASV19=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/wav2vec2-xls-r-300m/score_cm_eval.txt

#----------------------------------------------------------------#
#  ASVSPOOF5 PATHS
#----------------------------------------------------------------#
export ASV19_DEV_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac
export ASV19_DEV_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt
export ASV19_TRAIN_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac
export ASV19_TRAIN_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt
export ASV19_DEV_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac
export ASV19_DEV_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt

export MLAAD_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/multilingual
export MLAAD_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/multilingual/protocol_MLAAD_MAILabs_total_balanced.txt

export ASV5_TRAIN_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_train/flac
export ASV5_TRAIN_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5_train_protocol.txt
export ASV5_DEV_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_dev/flac
export ASV5_DEV_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.dev.metadata.txt
export ASV5_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac
export ASV5_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv

export ASV_DATASET=asv5
export ASV19_DATASET=asv19

echo "ASV5_TRAIN_ROOT=${ASV5_TRAIN_ROOT}"
echo "ASV5_TRAIN_PROTOCOL=${ASV5_TRAIN_PROTOCOL}"
echo "ASV5_DEV_ROOT=${ASV5_DEV_ROOT}"
echo "ASV5_DEV_PROTOCOL=${ASV5_DEV_PROTOCOL}"
echo "ASV5_EVAL_ROOT=${ASV5_EVAL_ROOT}"
echo "ASV5_EVAL_PROTOCOL=${ASV5_EVAL_PROTOCOL}"

#----------------------------------------------------------------#
#  RUN THE TRAINING SCRIPT
#----------------------------------------------------------------#
echo "EXPERIMENT NAME: ${EXP_NAME}"

cd ~/wav2vec_contr_loss

LAUNCHER="torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:${MASTER_PORT}"

echo "Using BATCH_SIZE=${BATCH_SIZE}"

export USE_QUEUE=${USE_QUEUE}
export QUEUE_SIZE=${QUEUE_SIZE}
export QUEUE_START_EPOCH=${QUEUE_START_EPOCH}
export CENTER_BEFORE_NORM=${CENTER_BEFORE_NORM}
export GLOBAL_CENTER=${GLOBAL_CENTER}

# ${LAUNCHER} train_stage1_asv5.py \
#   --model_name ${MODEL} \
#   --finetune_encoder ${FINETUNE_ENCODER} \
#   --max_duration_seconds ${MAX_DURATION_SECONDS} \
#   --save_dir ${CKPT_ROOT} \
#   --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} \
#   --supcon_similarity ${SUPCON_SIMILARITY} --uniformity_weight ${UNIFORMITY_WEIGHT} --uniformity_t ${UNIFORMITY_T} --sil_weight ${SIL_WEIGHT} \
#   --enc_lr ${ENC_LR} --head_lr ${HEAD_LR} --weight_decay ${WEIGHT_DECAY} --temperature ${TEMPERATURE} \
#   --num_workers ${NUM_WORKERS} --seed ${SEED} \
#   --topk_neg ${TOPK_NEG} --warmup_epochs ${WARMUP_EPOCHS} --alpha_end ${ALPHA_END} --alpha_ramp_epochs ${ALPHA_RAMP_EPOCHS} \
#   --use_rawboost ${USE_RAWBOOST} --rawboost_prob ${RAWBOOST_PROB} \
#   --use_bg_aug ${USE_BG_AUG}
# 
# if [ ! -f "${CKPT}" ]; then
#   echo "[ERROR] Stage-1 checkpoint not found: ${CKPT}"
#   exit 1
# fi

export SKIP_ITW=1

rm -rf ${ASV5_EMB_DIR} ${ASV19_EMB_DIR} ${MLAAD_EMB_DIR} ${POOLED_EMB_DIR} ${ITW_EMB_DIR}

# Extracting datasets

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

python extract_stage1_embeddings_generic.py \
  --dataset mlaad \
  --model_name "${MODEL}" \
  --stage1_ckpt "${CKPT}" \
  --out_dir "${MLAAD_EMB_DIR}" \
  --splits "train" \
  --train_root "${MLAAD_ROOT}" --train_protocol "${MLAAD_PROTOCOL}" \
  --max_duration_seconds "${MAX_DURATION_SECONDS}"

# Pooling dataset for stage 2

python pool_stage1_embeddings.py \
  --in_dirs "${ASV5_EMB_DIR},${ASV19_EMB_DIR},${MLAAD_EMB_DIR}" \
  --names "asv5,asv19,mlaad" \
  --out_dir "${POOLED_EMB_DIR}" \
  --splits "train,dev" \
  --allow_missing_splits

python train_stage2_classifier.py \
  --emb_dir "${POOLED_EMB_DIR}" \
  --save_dir "${STAGE2_DIR}" \
  --batch_size 32 --epochs 200 --lr 1e-4 --weight_decay 1e-4 \
  --head_type linear --hidden_dim 128 --dropout 0.2 --patience 15

if [ ! -f "${STAGE2_CKPT}" ]; then
  echo "[ERROR] Stage-2 checkpoint not found: ${STAGE2_CKPT}"
  exit 1
fi

python eval_datasets.py \
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
  --asv5_protocol ${ASV5_EVAL_PROTOCOL}

echo "----------------------------------------------------------------"
echo "Training script finished."
echo "Job finished at: $(date)"
