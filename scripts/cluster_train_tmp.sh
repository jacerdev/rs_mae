#!/usr/bin/env bash
#SBATCH --job-name=rs_mae_train

#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-gpu=12
#SBATCH --partition=gpu_short
#SBATCH --time=0-12:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=aloui@campus.tu-berlin.de
#SBATCH --output=/scratch/jacer/outputs/slurm/job-%j.slurm.out
#SBATCH --chdir=/home/users/j/jacer/projects/rs_mae/


# import SECRETS_PATH, get_value, send_telegram, on_exit
source /home/users/j/jacer/projects/rs_mae/scripts/functions.sh

trap on_exit EXIT TERM INT
mkdir -p '/scratch/jacer/outputs/slurm/'
mkdir -p '/scratch/jacer/outputs/training_out/' # Specified in configs/paths/cluster.yaml as wandb_outdir

set -e # Strict processing instructions

# Environment activation
export MAMBA_EXE='/home/users/j/jacer/programs/micromamba'
export MAMBA_ROOT_PREFIX='/home/users/j/jacer/.local/share/mamba'
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate /scratch/jacer/envs/rs_mae

# WandB
export WANDB_API_KEY=$(get_value "$SECRETS_PATH" "WANDB_API_KEY")
export WANDB_MODE=${WANDB_MODE:-online}

nvidia-smi || echo -e "\e[31mNo NVIDIA GPU available on this node\e[0m"
lscpu | grep -E "Model name|CPU\(s\):"
free -h
echo -e "• Node: $(hostname) • SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST\n• SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE\n• SLURM_GTIDS: $SLURM_GTIDS"
echo "Using python: $(which python)"



send_telegram "Finetuning baseline"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 1: reben_7k"
python -m train.finetune experiment_name="baseline" segmentation_ds=reben_7k ckpt_export_path=best_ckpt_baseline.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 2: sen1floods11"
python -m train.finetune experiment_name="baseline" segmentation_ds=sen1floods11 ckpt_export_path=best_ckpt_baseline.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune experiment_name="baseline" segmentation_ds=cashew_plantation ckpt_export_path=best_ckpt_baseline.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 4: c2seg"
python -m train.finetune experiment_name="baseline" segmentation_ds=c2seg ckpt_export_path=best_ckpt_baseline.txt "$@" &
child_pid=$!
wait "$child_pid"

send_telegram "Finetuning feat_guided"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 1: reben_7k"
python -m train.finetune experiment_name="feat_guided" segmentation_ds=reben_7k ckpt_export_path=best_ckpt_feat_guided.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 2: sen1floods11"
python -m train.finetune experiment_name="feat_guided" segmentation_ds=sen1floods11 ckpt_export_path=best_ckpt_feat_guided.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune experiment_name="feat_guided" segmentation_ds=cashew_plantation ckpt_export_path=best_ckpt_feat_guided.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 4: c2seg"
python -m train.finetune experiment_name="feat_guided" segmentation_ds=c2seg ckpt_export_path=best_ckpt_feat_guided.txt "$@" &
child_pid=$!
wait "$child_pid"

send_telegram "Finetuning curriculum_ndi"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 1: reben_7k"
python -m train.finetune experiment_name="curriculum_ndi" segmentation_ds=reben_7k ckpt_export_path=best_ckpt_curriculum_ndi.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 2: sen1floods11"
python -m train.finetune experiment_name="curriculum_ndi" segmentation_ds=sen1floods11 ckpt_export_path=best_ckpt_curriculum_ndi.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune experiment_name="curriculum_ndi" segmentation_ds=cashew_plantation ckpt_export_path=best_ckpt_curriculum_ndi.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 4: c2seg"
python -m train.finetune experiment_name="curriculum_ndi" segmentation_ds=c2seg ckpt_export_path=best_ckpt_curriculum_ndi.txt "$@" &
child_pid=$!
wait "$child_pid"

send_telegram "Finetuning fourier_loss"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 1: reben_7k"
python -m train.finetune experiment_name="fourier_loss" segmentation_ds=reben_7k ckpt_export_path=best_ckpt_fourier_loss.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 2: sen1floods11"
python -m train.finetune experiment_name="fourier_loss" segmentation_ds=sen1floods11 ckpt_export_path=best_ckpt_fourier_loss.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune experiment_name="fourier_loss" segmentation_ds=cashew_plantation ckpt_export_path=best_ckpt_fourier_loss.txt "$@" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 4: c2seg"
python -m train.finetune experiment_name="fourier_loss" segmentation_ds=c2seg ckpt_export_path=best_ckpt_fourier_loss.txt "$@" &
child_pid=$!
wait "$child_pid"

