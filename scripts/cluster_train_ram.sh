#!/usr/bin/env bash
#SBATCH --job-name=rs_mae_train

#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250G
#SBATCH --cpus-per-gpu=20
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

# --- DATA PREPARATION (RAM DISK) ---
LOCAL_RAM="/dev/shm/jacer/datasets/reben"
REMOTE_DATA="/scratch/jacer/datasets/reben/BigEarthNet-S2_lmdb"
if [ ! -d "$LOCAL_RAM/BigEarthNet-S2_lmdb" ]; then
    echo "[LOG] Copying dataset to RAM Disk ($LOCAL_RAM)..."
    mkdir -p "$LOCAL_RAM"
    rsync -ah "$REMOTE_DATA" "$LOCAL_RAM/"
    echo "[LOG] Dataset copied successfully."
else
    echo "[LOG] Dataset already present in RAM Disk."
fi

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

if [ -z "$1" ]; then
    echo "Usage: sbatch $0 <experiment_name> [hydra_overrides...]"
    exit 1
fi
experiment=$1
shift

send_telegram "Experiment: $experiment"

echo "--------------------------------------------------"
send_telegram "🚀 Pretraining [$(date "+%Y-%m-%d %H:%M:%S")]"
python -m train.pretrain "$@" experiment="$experiment" paths.fast_dir="/dev/shm" &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 1: reben_7k"
python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=reben_7k &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 2: sen1floods11"
python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=sen1floods11 &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=cashew_plantation &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 4: c2seg"
python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=c2seg &
child_pid=$!
wait "$child_pid"
# echo "--------------------------------------------------"
# send_telegram "🚀 Finetuning 5: burn_scars"
# python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=burn_scars &
# child_pid=$!
# wait "$child_pid"
