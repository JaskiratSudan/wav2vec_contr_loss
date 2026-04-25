#!/bin/bash
#SBATCH --job-name=mlaad_mem_bank_16384_supcon_geodesic_temp_0.05
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
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

#----------------------------------------------------------------#
#  EXPERIMENT VARIABLES
#----------------------------------------------------------------#
EXP_NAME=mlaad_mem_bank_16384_supcon_geodesic_temp_0.05

# SUPCON
SUPCON_SIMILARITY=geodesic
TEMPERATURE=0.05

# UNIFORMITY TERM
UNIFORMITY_WEIGHT=0.00
UNIFORMITY_T=0.5

# HARD NEGATIVE MINING 
TOPK_NEG=256
WARMUP_EPOCHS=100
ALPHA_END=0.9
ALPHA_RAMP_EPOCHS=20

# MEMORY BANK
USE_QUEUE=1
export QUEUE_SIZE=16384

# PREVENT COLLAPSE OF SUPCON
CENTER_BEFORE_NORM=0
GLOBAL_CENTER=0

# MULTI GPU SETUP
export ACCUM_STEPS=2

export MLAAD_STAGE2_SPLIT_SEED=2027
export MLAAD_MAX_DURATION_SECONDS=10
export USE_AMP=1


MODEL_PATH=facebook/wav2vec2-xls-r-300m
MODEL=wav2vec2-xls-r-300m

CKPT_ROOT=/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1/${EXP_NAME}
CKPT=${CKPT_ROOT}/${MODEL}_stage1_head_best.pt

ASV19_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV19/${EXP_NAME}
ASV5_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV5/${EXP_NAME}
ITW_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/${EXP_NAME}
FF_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/FamousFigures/${EXP_NAME}
FAKEXPOSE_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/FakeXpose/${EXP_NAME}
MLAAD_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/MLAAD/${EXP_NAME}

STAGE2_DIR=checkpoints_stage2/${EXP_NAME}/${MODEL}
STAGE2_CKPT=${STAGE2_DIR}/stage2_binary_head_best.pt

SCORE_ASV19=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/${MODEL}/score_cm_eval.txt
SCORE_ASV5=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/${MODEL}/score_cm_eval_asv5.txt
SCORE_ITW=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/${MODEL}/score_cm_itw.txt
SCORE_FF=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/famous_figures/score_cm_ff.txt
SCORE_FAKEXPOSE=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/fakexpose/score_cm_fakexpose.txt
SCORE_MLAAD=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/mlaad/score_cm_mlaad_dev.txt

#----------------------------------------------------------------#
#  MLAAD/MAILABS PATHS
#----------------------------------------------------------------#
export MLAAD_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/multilingual
export MLAAD_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/multilingual/protocol_MLAAD_MAILabs_total_balanced.txt


ASV19_TRAIN_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac
ASV19_TRAIN_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt
ASV19_DEV_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac
ASV19_DEV_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt
ASV19_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac
ASV19_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt

ASV5_TRAIN_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_train/flac
ASV5_TRAIN_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5_train_protocol.txt
ASV5_DEV_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_dev/flac
ASV5_DEV_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.dev.metadata.txt
ASV5_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac
ASV5_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5_eval_metadata.txt

ITW_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild
ITW_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv

FF_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt
FF_ROOT=

FAKEXPOSE_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/fakexpose

echo "MLAAD_ROOT=${MLAAD_ROOT}"
echo "MLAAD_PROTOCOL=${MLAAD_PROTOCOL}"
echo "MLAAD_STAGE2_SPLIT_SEED=${MLAAD_STAGE2_SPLIT_SEED}"
echo "MLAAD_MAX_DURATION_SECONDS=${MLAAD_MAX_DURATION_SECONDS}"
echo "USE_AMP=${USE_AMP}"
echo "ACCUM_STEPS=${ACCUM_STEPS}"
echo "USE_QUEUE"=${USE_QUEUE}
echo "QUEUE_SIZE=${QUEUE_SIZE}"

#----------------------------------------------------------------#
#  RUN THE TRAINING SCRIPT
#----------------------------------------------------------------#
echo "EXPERIMENT NAME: ${EXP_NAME}"

cd ~/wav2vec_contr_loss

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
  NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
else
  NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
fi
echo "Detected NUM_GPUS=${NUM_GPUS}"

export CENTER_BEFORE_NORM=${CENTER_BEFORE_NORM}
export GLOBAL_CENTER=${GLOBAL_CENTER}
export USE_QUEUE=${USE_QUEUE}
export QUEUE_SIZE=${QUEUE_SIZE}
export USE_AMP=${USE_AMP}
export ACCUM_STEPS=${ACCUM_STEPS}

LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --rdzv_endpoint=localhost:${MASTER_PORT}"
BATCH_SIZE=32
EXTRACTION_BATCH_SIZE=32

echo "Using BATCH_SIZE=${BATCH_SIZE}"

${LAUNCHER} train_stage1_mlaad.py \
  --model_name ${MODEL_PATH} \
  --finetune_encoder 1 \
  --save_dir ${CKPT_ROOT} \
  --epochs 100 --batch_size ${BATCH_SIZE} --num_samples none \
  --supcon_similarity ${SUPCON_SIMILARITY} --uniformity_weight 0.00 --uniformity_t 0.5 \
  --enc_lr 1e-5 --head_lr 5e-3 --weight_decay 3e-3 --temperature 0.05 \
  --num_workers 6 --seed 1337 \
  --topk_neg 15 --warmup_epochs 100 --alpha_end 1 --alpha_ramp_epochs 80 \
  --use_rawboost 1 --rawboost_prob 0.7 \
  --use_bg_aug 1

TRAIN_EXIT=$?
if [ ${TRAIN_EXIT} -ne 0 ]; then
  echo "[ERROR] Stage-1 training failed (exit=${TRAIN_EXIT}). Skipping extraction/eval."
  exit ${TRAIN_EXIT}
fi

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${CKPT}"
  exit 1
fi

rm -rf ${ASV19_EMB_DIR}
rm -rf ${ASV5_EMB_DIR}
rm -rf ${ITW_EMB_DIR}
rm -rf ${FF_EMB_DIR}
rm -rf ${FAKEXPOSE_EMB_DIR}
rm -rf ${MLAAD_EMB_DIR}

python extract_stage1_embeddings_all.py \
  --model_name ${MODEL_PATH} \
  --stage1_ckpt ${CKPT} \
  --batch_size ${EXTRACTION_BATCH_SIZE} --num_workers 6 --seed 1337 \
  --max_duration_seconds ${MLAAD_MAX_DURATION_SECONDS} \
  --mlaad_root ${MLAAD_ROOT} \
  --mlaad_protocol ${MLAAD_PROTOCOL} \
  --mlaad_out_dir ${MLAAD_EMB_DIR} \
  --mlaad_split_seed ${MLAAD_STAGE2_SPLIT_SEED}

python train_stage2_classifier.py \
  --emb_dir ${MLAAD_EMB_DIR} \
  --save_dir ${STAGE2_DIR} \
  --batch_size 64 --epochs 200 --lr 1e-4 --weight_decay 1e-4 \
  --head_type linear --hidden_dim 128 --dropout 0.2 --patience 15

python generate_score_files_all.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --mlaad_dir ${MLAAD_EMB_DIR} \
  --score_mlaad ${SCORE_MLAAD}

# ITW
python extract_stage1_embeddings_all.py \
  --model_name ${MODEL_PATH} \
  --stage1_ckpt ${CKPT} \
  --batch_size ${EXTRACTION_BATCH_SIZE} --num_workers 6 --seed 1337 \
  --max_duration_seconds ${MLAAD_MAX_DURATION_SECONDS} \
  --itw_root ${ITW_ROOT} \
  --itw_protocol ${ITW_PROTOCOL} \
  --itw_out_dir ${ITW_EMB_DIR}

python generate_score_files_all.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --itw_dir ${ITW_EMB_DIR} \
  --score_itw ${SCORE_ITW}

python eval_score_files.py --score ITW=${SCORE_ITW}

# ASV19
python extract_stage1_embeddings_all.py \
  --model_name ${MODEL_PATH} \
  --stage1_ckpt ${CKPT} \
  --batch_size ${EXTRACTION_BATCH_SIZE} --num_workers 6 --seed 1337 \
  --max_duration_seconds ${MLAAD_MAX_DURATION_SECONDS} \
  --asv19_train_root ${ASV19_TRAIN_ROOT} \
  --asv19_train_protocol ${ASV19_TRAIN_PROTOCOL} \
  --asv19_dev_root ${ASV19_DEV_ROOT} \
  --asv19_dev_protocol ${ASV19_DEV_PROTOCOL} \
  --asv19_eval_root ${ASV19_EVAL_ROOT} \
  --asv19_eval_protocol ${ASV19_EVAL_PROTOCOL} \
  --asv19_out_dir ${ASV19_EMB_DIR}

python generate_score_files_all.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --asv19_dir ${ASV19_EMB_DIR} \
  --score_asv19 ${SCORE_ASV19}

python eval_score_files.py --score ASV19=${SCORE_ASV19}

# FakeXpose
python extract_stage1_embeddings_all.py \
  --model_name ${MODEL_PATH} \
  --stage1_ckpt ${CKPT} \
  --batch_size ${EXTRACTION_BATCH_SIZE} --num_workers 6 --seed 1337 \
  --max_duration_seconds ${MLAAD_MAX_DURATION_SECONDS} \
  --fakexpose_root ${FAKEXPOSE_ROOT} \
  --fakexpose_out_dir ${FAKEXPOSE_EMB_DIR}

python generate_score_files_all.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --fakexpose_dir ${FAKEXPOSE_EMB_DIR} \
  --score_fakexpose ${SCORE_FAKEXPOSE}

python eval_score_files.py --score FakeXpose=${SCORE_FAKEXPOSE}

# Famous Figures
python extract_stage1_embeddings_all.py \
  --model_name ${MODEL_PATH} \
  --stage1_ckpt ${CKPT} \
  --batch_size ${EXTRACTION_BATCH_SIZE} --num_workers 6 --seed 1337 \
  --max_duration_seconds ${MLAAD_MAX_DURATION_SECONDS} \
  --ff_protocol ${FF_PROTOCOL} \
  --ff_root "${FF_ROOT}" \
  --ff_out_dir ${FF_EMB_DIR}

python generate_score_files_all.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --ff_dir ${FF_EMB_DIR} \
  --score_ff ${SCORE_FF}

python eval_score_files.py --score FamousFigures=${SCORE_FF}

# ASV5
python extract_stage1_embeddings_all.py \
  --model_name ${MODEL_PATH} \
  --stage1_ckpt ${CKPT} \
  --batch_size ${EXTRACTION_BATCH_SIZE} --num_workers 6 --seed 1337 \
  --max_duration_seconds ${MLAAD_MAX_DURATION_SECONDS} \
  --asv5_train_root ${ASV5_TRAIN_ROOT} \
  --asv5_train_protocol ${ASV5_TRAIN_PROTOCOL} \
  --asv5_dev_root ${ASV5_DEV_ROOT} \
  --asv5_dev_protocol ${ASV5_DEV_PROTOCOL} \
  --asv5_eval_root ${ASV5_EVAL_ROOT} \
  --asv5_eval_protocol ${ASV5_EVAL_PROTOCOL} \
  --asv5_out_dir ${ASV5_EMB_DIR}

python generate_score_files_all.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --asv5_dir ${ASV5_EMB_DIR} \
  --score_asv5 ${SCORE_ASV5}

python eval_score_files.py --score ASV5=${SCORE_ASV5}

python eval_score_files.py \
  --score ITW=${SCORE_ITW} \
  --score ASV19=${SCORE_ASV19} \
  --score FakeXpose=${SCORE_FAKEXPOSE} \
  --score FamousFigures=${SCORE_FF} \
  --score ASV5=${SCORE_ASV5} \
  --score MLAAD=${SCORE_MLAAD}

echo "----------------------------------------------------------------"
echo "Training script finished."
echo "Job finished at: $(date)"
