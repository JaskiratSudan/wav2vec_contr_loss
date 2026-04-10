#!/bin/bash
#SBATCH --job-name=stage1_plots
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=plot_logs/%x-%j.log

module purge
module load python/3.9.12
module load cuda/11.8.0
module load ffmpeg
source ~/myenv/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "------------------------------------------------"

# ==========================================
# EDIT THIS LIST ONLY
# ==========================================
EXPERIMENTS=(
    # Figure A — temperature sweep, no queue
    # supcon_temp_0.07
    # supcon_temp_0.1
    # supcon_temp_0.3
    # supcon_temp_0.6
    # supcon_geodesic_temp_0.07
    # supcon_geodesic_temp_0.1
    # supcon_geodesic_temp_0.3
    # supcon_geodesic_temp_0.6

    # Figure B — queue ablation, best temp per similarity
    # mem_256_supcon_temp_0.3
    # mem_256_supcon_geodesic_temp_0.07
    # mem_1024_supcon_temp_0.3
    # mem_1024_supcon_geodesic_temp_0.07
    mem_4096_supcon_temp_0.3
    mem_4096_supcon_geodesic_temp_0.07
    mem_8192_supcon_temp_0.3
    mem_8192_supcon_geodesic_temp_0.07
)

# ==========================================
MODEL_PATH=facebook/wav2vec2-xls-r-300m
MODEL_TAG=facebook__wav2vec2-xls-r-300m
PLOT_TYPE=tsne
# ==========================================

for EXP_NAME in "${EXPERIMENTS[@]}"; do
    echo "==============================================="
    echo "Running plots for: ${EXP_NAME}"
    echo "==============================================="

    CKPT=/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1/${EXP_NAME}/wav2vec2-xls-r-300m_stage1_head_best.pt

    if [ ! -f "${CKPT}" ]; then
        echo "[WARNING] Checkpoint not found for ${EXP_NAME}, skipping..."
        continue
    fi

    python ~/wav2vec_contr_loss/plot_stage1_umap_itw.py \
        --model_name ${MODEL_PATH} \
        --ckpt_path ${CKPT} \
        --plots_dir ~/wav2vec_contr_loss/plots/dep_embeddings/ITW/${EXP_NAME} \
        --exp_name ${EXP_NAME} \
        --dr_method ${PLOT_TYPE} \
        --clean

done

echo "------------------------------------------------"
echo "All plotting finished at $(date)"