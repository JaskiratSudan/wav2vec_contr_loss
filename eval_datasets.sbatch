#!/bin/bash
#SBATCH --job-name=eval_datasets
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=eval_datasets_log/%x-%j.log

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

#----------------------------------------------------------------#
#  EXPERIMENT VARIABLES (EDIT THESE)
#----------------------------------------------------------------#
EXP_NAMES=(
  # supcon_temp_0.07
  # supcon_temp_0.1
  # supcon_temp_0.3
  # supcon_temp_0.6
  # supcon_geodesic_temp_0.07
  # supcon_geodesic_temp_0.1
  # supcon_geodesic_temp_0.3
  # supcon_geodesic_temp_0.6
  # mem_256_supcon_temp_0.07
  # mem_256_supcon_temp_0.3
  # mem_256_supcon_geodesic_temp_0.07
  # mem_256_supcon_geodesic_temp_0.3
  # mem_256_supcon_temp_0.07_uni_weight_0.1_uni_t_2
  # mem_256_supcon_temp_0.3_uni_weight_0.1_uni_t_2
  # mem_256_supcon_geodesic_temp_0.07_uni_weight_0.1_uni_t_2
  # mem_256_supcon_geodesic_temp_0.3_uni_weight_0.1_uni_t_2
  # mem_256_qst_12_supcon_temp_0.07
  # mem_256_qst_12_supcon_temp_0.3
  # mem_256_qst_12_supcon_geodesic_temp_0.07
  # mem_256_qst_12_supcon_geodesic_temp_0.3
  # mem_256_qst_12_supcon_temp_0.07_uni_weight_0.1_uni_t_2
  # mem_256_qst_12_supcon_temp_0.3_uni_weight_0.1_uni_t_2
  # mem_256_qst_12_supcon_geodesic_temp_0.07_uni_weight_0.1_uni_t_2
  # mem_256_qst_12_supcon_geodesic_temp_0.3_uni_weight_0.1_uni_t_2
  # mem_1024_supcon_temp_0.07
  # mem_1024_supcon_temp_0.3
  # mem_1024_supcon_geodesic_temp_0.07
  # mem_1024_supcon_geodesic_temp_0.3
  # mem_4096_supcon_temp_0.07
  # mem_4096_supcon_temp_0.3
  # mem_4096_supcon_geodesic_temp_0.07
  # mem_4096_supcon_geodesic_temp_0.3
  # supcon_geodesic_temp_0.07
  # mem_256_supcon_geodesic_temp_0.07
  # mem_16384_supcon_temp_0.3
  # asv5_supcon_geodesic_temp_0.07
  # asv5_supcon_geodesic_temp_0.07
  # asv5_mem_256_supcon_geodesic_temp_0.07
  # asv5_hbm_1024_mem_16384_supcon_geodesic_temp_0.07
  # asv5_mem_4096_qst_3_supcon_temp_0.3
  # asv5_mem_4096_qst_3_supcon_geodesic_temp_0.07
  asv5_hbm_1024_mem_16384_supcon_geodesic_temp_0.07
)

MODEL=facebook/wav2vec2-xls-r-300m
DATASET_LIST=(
  asv19
  itw
  fakexpose
  deepfake_eval_2024
  asv21_la
  # famous_figures
  asv5
  mlaad
  # asv21_df
)

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

WINDOWED_EVAL=0
WINDOW_SEC=5
HOP_SEC=5
AGG=mean
TOPK_FRAC=0.3
MAX_WINDOWS_PER_UTT=0
WINDOW_BATCH_SIZE=256

DATASETS=$(IFS=,; echo "${DATASET_LIST[*]}")
SUBSET=all

BATCH_SIZE=64
NUM_WORKERS=8
MAX_DURATION_SECONDS=10
TARGET_SAMPLE_RATE=16000

#----------------------------------------------------------------#
#  RUN EVALUATION
#----------------------------------------------------------------#
cd ~/wav2vec_contr_loss || exit 1

for EXP_NAME in "${EXP_NAMES[@]}"; do
  echo "----------------------------------------------------------------"
  echo "EXPERIMENT NAME: ${EXP_NAME}"

  # Delete any existing score files for this experiment
  # rm -rf "/home/jsudan/wav2vec_contr_loss/scores/${EXP_NAME}"
  
  python eval_datasets.py \
    --exp_name "${EXP_NAME}" \
    --datasets "${DATASETS}" \
    --model_name "${MODEL}" \
    --stage1_root "/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1"\
    --subset "${SUBSET}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_duration_seconds "${MAX_DURATION_SECONDS}" \
    --target_sample_rate "${TARGET_SAMPLE_RATE}" \
    --ff_speakers "${FF_SPEAKERS}" \
    --print_eer \
    --windowed_eval ${WINDOWED_EVAL} \
    --window_sec ${WINDOW_SEC} \
    --hop_sec ${HOP_SEC} \
    --agg ${AGG} \
    --topk_frac ${TOPK_FRAC} \
    --max_windows_per_utt ${MAX_WINDOWS_PER_UTT} \
    --window_batch_size ${WINDOW_BATCH_SIZE}
done

echo "----------------------------------------------------------------"
echo "Evaluation finished at: $(date)"
