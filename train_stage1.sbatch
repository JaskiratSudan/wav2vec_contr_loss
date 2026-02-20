#!/bin/bash
#SBATCH --job-name=mem_bank_128_supcon_temp_0.05_uni_weight_0.05_uni_t_0.5  # Job name
#SBATCH --partition=spgpu                # GPU partition
#SBATCH --account=hafiz1               # Great Lakes account name
#SBATCH --gpus-per-node=2              # GPU per node
#SBATCH --mem=36G                      # CPU RAM
#SBATCH --cpus-per-task=4             # CPU cores 
#SBATCH --time=02:00:00                # time requested (hh:mm:ss)
#SBATCH --output=train_stage1_log/%x-%j.log             # (%x=job_name, %j=job_id)

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

# Avoid DDP port collisions when multiple jobs run on the same node.
export MASTER_PORT=$((29500 + $SLURM_JOB_ID % 1000))

#----------------------------------------------------------------#
#  EXPERIMENT VARIABLES
#----------------------------------------------------------------#
EXP_NAME=mem_bank_128_supcon_temp_0.05_uni_weight_0.05_uni_t_0.5
SUPCON_SIMILARITY=cosine
MODEL_PATH=facebook/wav2vec2-xls-r-300m
MODEL=wav2vec2-xls-r-300m
USE_RAVDESS=0
USE_COMMONVOICE=0
USE_QUEUE=1
QUEUE_SIZE=128
ACCUM_STEPS=1
CENTER_BEFORE_NORM=1
GLOBAL_CENTER=1

CKPT_ROOT=/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1/${EXP_NAME}
CKPT=${CKPT_ROOT}/${MODEL}_stage1_head_best.pt

ASV_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV/${EXP_NAME}
ITW_EMB_DIR=/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/${EXP_NAME}

STAGE2_DIR=checkpoints_stage2/${EXP_NAME}/${MODEL}
STAGE2_CKPT=${STAGE2_DIR}/stage2_binary_head_best.pt

SCORE_ASV=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/${MODEL}/score_cm_eval.txt
SCORE_ITW=/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}/${MODEL}/score_cm_itw.txt

#----------------------------------------------------------------#
#  RUN THE TRAINING SCRIPT
#----------------------------------------------------------------#
echo "EXPERIMENT NAME: ${EXP_NAME}"
echo "USE_QUEUE=${USE_QUEUE}"
echo "QUEUE_SIZE=${QUEUE_SIZE}"
echo "ACCUM_STEPS=${ACCUM_STEPS}"
echo "CENTER_BEFORE_NORM=${CENTER_BEFORE_NORM}"
echo "GLOBAL_CENTER=${GLOBAL_CENTER}"

# Navigate to your project directory if the script is not in your home directory
cd ~/wav2vec_contr_loss

# Execute your main training script
# python train_stage1.py --model_name facebook/wav2vec2-large-960h
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
export CENTER_BEFORE_NORM=${CENTER_BEFORE_NORM}
export GLOBAL_CENTER=${GLOBAL_CENTER}

# ${LAUNCHER} train_stage1.py \
#   --model_name ${MODEL_PATH} \
#   --finetune_encoder 1 \
#   --save_dir ${CKPT_ROOT} \
#   --epochs 100 --batch_size ${BATCH_SIZE} --num_samples none \
#   --accum_steps ${ACCUM_STEPS} \
#   --supcon_similarity ${SUPCON_SIMILARITY} --uniformity_weight 0.0 --uniformity_t 0.5 \
#   --enc_lr 1e-5 --head_lr 5e-3 --weight_decay 3e-3 --temperature 0.05 \
#   --num_workers 4 --seed 1337 \
#   --topk_neg 15 --warmup_epochs 100 --alpha_end 1 --alpha_ramp_epochs 80 \
#   --use_rawboost 1 --rawboost_prob 0.7 \
#   --use_ravdess ${USE_RAVDESS} --use_commonvoice ${USE_COMMONVOICE}

# if [ ! -f "${CKPT}" ]; then
#   echo "[ERROR] Stage-1 checkpoint not found: ${CKPT}"
#   exit 1
# fi

# python plot_stage1_umap_asv.py \
#   --model_name ${MODEL_PATH} \
#   --ckpt_path ${CKPT} \
#   --plots_dir /home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ASV/${EXP_NAME} \
#   --exp_name ${EXP_NAME}
# python plot_stage1_umap_itw.py \
#   --model_name ${MODEL_PATH} \
#   --ckpt_path ${CKPT} \
#   --plots_dir /home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ITW/${EXP_NAME} \
#   --exp_name ${EXP_NAME}

rm -rf ${ASV_EMB_DIR}
rm -rf ${ITW_EMB_DIR}

python extract_stage1_embeddings.py \
  --model_name ${MODEL_PATH} \
  --stage1_ckpt ${CKPT} \
  --asv_out_dir ${ASV_EMB_DIR} \
  --itw_out_dir ${ITW_EMB_DIR}

python train_stage2_classifier.py \
  --emb_dir ${ASV_EMB_DIR} \
  --save_dir ${STAGE2_DIR} \
  --batch_size 64 --epochs 200 --lr 1e-4 --weight_decay 1e-4 \
  --head_type linear --hidden_dim 128 --dropout 0.2 --patience 15

rm -rf ${SCORE_ASV}
rm -rf ${SCORE_ITW}

python generate_eval_score_file.py \
  --stage2_ckpt ${STAGE2_CKPT} \
  --eval_emb_path ${ASV_EMB_DIR}/eval_embeddings.npy \
  --eval_label_path ${ASV_EMB_DIR}/eval_labels.npy \
  --itw_emb_path ${ITW_EMB_DIR}/itw_embeddings.npy \
  --itw_label_path ${ITW_EMB_DIR}/itw_labels.npy \
  --score_file_asv_eval ${SCORE_ASV} \
  --score_file_itw ${SCORE_ITW} \
  --batch_size 256 --run_asv_eval 1 --run_itw 1

echo "----------------------------------------------------------------"
echo "Training script finished."
echo "Job finished at: $(date)"
