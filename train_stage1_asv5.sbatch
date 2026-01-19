#!/bin/bash
#SBATCH --job-name=asv5_windowed_supcon_geodesic_temp_0.05
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=4
#SBATCH --mem=36G
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=train_stage1_log/%x-%j.log

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

export MASTER_PORT=$((29500 + $SLURM_JOB_ID % 1000))

#----------------------------------------------------------------#
#  EXPERIMENT VARIABLES
#----------------------------------------------------------------#
EXP_NAME=asv5_windowed_supcon_geodesic_temp_0.05
SUPCON_SIMILARITY=geodesic
MODEL=facebook/wav2vec2-xls-r-300m
RUN_TAG=${MODEL//\//__}

CKPT_ROOT=/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1/${EXP_NAME}
CKPT_DIR=${CKPT_ROOT}/${RUN_TAG}
CKPT=${CKPT_ROOT}/${RUN_TAG}/${RUN_TAG}_stage1_head_best.pt

ASV5_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV5/${EXP_NAME}
ASV19_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV19/${EXP_NAME}
ITW_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/${EXP_NAME}

STAGE2_DIR=checkpoints_stage2/${EXP_NAME}/facebook/wav2vec2-xls-r-300m
STAGE2_CKPT=${STAGE2_DIR}/stage2_binary_head_best.pt

SCORE_ASV=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/facebook/wav2vec2-xls-r-300m/score_cm_eval_asv5.txt
SCORE_ITW=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/facebook/wav2vec2-xls-r-300m/score_cm_itw.txt
SCORE_ASV19=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/facebook/wav2vec2-xls-r-300m/score_cm_eval.txt

#----------------------------------------------------------------#
#  ASVSPOOF5 PATHS
#----------------------------------------------------------------#
export ASV5_TRAIN_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_train/flac
export ASV5_TRAIN_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5_train_protocol.txt
export ASV5_DEV_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_dev/flac
export ASV5_DEV_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.dev.metadata.txt
export ASV5_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac
export ASV5_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5_eval_metadata.txt
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

LAUNCHER="torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:${MASTER_PORT}"
BATCH_SIZE=32

echo "Using BATCH_SIZE=${BATCH_SIZE}"

${LAUNCHER} train_stage1_asv5.py \
  --model_name ${MODEL} \
  --finetune_encoder 1 \
  --save_dir ${CKPT_ROOT} \
  --epochs 100 --batch_size ${BATCH_SIZE} --num_samples none \
  --supcon_similarity ${SUPCON_SIMILARITY} --uniformity_weight 0.0 --uniformity_t 2.0 \
  --enc_lr 1e-5 --head_lr 5e-3 --weight_decay 3e-3 --temperature 0.05 \
  --num_workers 6 --seed 1337 \
  --topk_neg 15 --warmup_epochs 100 --alpha_end 1 --alpha_ramp_epochs 80 \
  --use_rawboost 0 --rawboost_prob 0.0

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${CKPT}"
  exit 1
fi

rm -rf ${ASV5_EMB_DIR}
rm -rf ${ASV19_EMB_DIR}
rm -rf ${ITW_EMB_DIR}

ASV_DATASET=${ASV_DATASET} python extract_stage1_embeddings.py \
  --model_name ${MODEL} \
  --stage1_ckpt ${CKPT} \
  --asv_out_dir ${ASV5_EMB_DIR} \
  --itw_out_dir ${ITW_EMB_DIR}

ASV_DATASET=${ASV19_DATASET} python extract_stage1_embeddings.py \
  --model_name ${MODEL} \
  --stage1_ckpt ${CKPT} \
  --asv_out_dir ${ASV19_EMB_DIR} \
  --itw_out_dir ${ITW_EMB_DIR}

python train_stage2_classifier.py \
  --emb_dir ${ASV5_EMB_DIR} \
  --save_dir ${STAGE2_DIR} \
  --batch_size 64 --epochs 200 --lr 1e-4 --weight_decay 1e-4 \
  --head_type linear --hidden_dim 128 --dropout 0.2 --patience 15

python generate_eval_score_file.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --eval_emb_path ${ASV5_EMB_DIR}/eval_embeddings.npy \
  --eval_label_path ${ASV5_EMB_DIR}/eval_labels.npy \
  --itw_emb_path ${ITW_EMB_DIR}/itw_embeddings.npy \
  --itw_label_path ${ITW_EMB_DIR}/itw_labels.npy \
  --score_file_asv_eval ${SCORE_ASV} \
  --score_file_itw ${SCORE_ITW} \
  --batch_size 256 --run_asv_eval 1 --run_itw 1

python generate_eval_score_file.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --eval_emb_path ${ASV19_EMB_DIR}/eval_embeddings.npy \
  --eval_label_path ${ASV19_EMB_DIR}/eval_labels.npy \
  --itw_emb_path ${ITW_EMB_DIR}/itw_embeddings.npy \
  --itw_label_path ${ITW_EMB_DIR}/itw_labels.npy \
  --score_file_asv_eval ${SCORE_ASV19} \
  --score_file_itw ${SCORE_ITW} \
  --batch_size 256 --run_asv_eval 1 --run_itw 1

python eval_score_files.py \
  --score ASV5=${SCORE_ASV} \
  --score ASV19=${SCORE_ASV19} \
  --score ITW=${SCORE_ITW}

echo "----------------------------------------------------------------"
echo "Training script finished."
echo "Job finished at: $(date)"
