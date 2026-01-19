#!/bin/bash
#SBATCH --job-name=mlaad_umap_plots
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=plot_stage1_umap_log/%x-%j.log

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

EXP_NAME=mlaad_windowed_supcon_geodesic_temp_0.05
MODEL=facebook/wav2vec2-xls-r-300m
RUN_TAG=${MODEL//\//__}

CKPT=/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1/${EXP_NAME}/${RUN_TAG}/${RUN_TAG}_stage1_head_best.pt

PLOTS_ASV=/home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ASV/${EXP_NAME}
PLOTS_ITW=/home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ITW/${EXP_NAME}
PLOTS_ASV5=/home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ASV5/${EXP_NAME}

ASV5_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac
ASV5_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5_eval_metadata.txt

cd ~/wav2vec_contr_loss || exit 1

python plot_stage1_umap_asv.py \
  --model_name ${MODEL} \
  --ckpt_path ${CKPT} \
  --plots_dir ${PLOTS_ASV}

python plot_stage1_umap_itw.py \
  --model_name ${MODEL} \
  --ckpt_path ${CKPT} \
  --plots_dir ${PLOTS_ITW}

python plot_stage1_umap_asv5_simple.py \
  --model_name ${MODEL} \
  --ckpt_path ${CKPT} \
  --plots_dir ${PLOTS_ASV5} \
  --eval_root ${ASV5_EVAL_ROOT} \
  --eval_protocol ${ASV5_EVAL_PROTOCOL}

echo "----------------------------------------------------------------"
echo "Plotting finished at: $(date)"
