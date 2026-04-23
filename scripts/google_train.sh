#!/usr/bin/env bash

# import SECRETS_PATH, get_value, send_telegram, on_exit
source scripts/functions.sh

# e.g: CLOUD_PREFIX="/content/drive/MyDrive/rs_mae" bash scripts/google_train.sh "baseline"
CLOUD_PREFIX=${CLOUD_PREFIX:-"cloud"}

export SECRETS_PATH="$CLOUD_PREFIX/personal.env"

trap on_exit EXIT TERM INT
mkdir -p "$CLOUD_PREFIX/outputs/training_out/" # Specified in configs/paths/google.yaml as wandb_outdir

set -e # Strict processing instructions

# Environment activation
export MAMBA_EXE="$CLOUD_PREFIX/programs/micromamba/bin/micromamba"
#export MAMBA_ROOT_PREFIX="$CLOUD_PREFIX/envs"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate "$CLOUD_PREFIX/envs/rs_mae"

# WandB
export WANDB_API_KEY=$(get_value "$SECRETS_PATH" "WANDB_API_KEY")
export WANDB_MODE=${WANDB_MODE:-online}

nvidia-smi || echo -e "\e[31mNo NVIDIA GPU available on this node\e[0m"
lscpu | grep -E "Model name|CPU\(s\):"
free -h
echo -e "• Node: $(hostname)"
echo "Using python: $(which python)"
echo "Using cloud boundary prefix: $CLOUD_PREFIX"

if [ -z "$1" ]; then
    echo "Usage: CLOUD_PREFIX='/my/path' bash $0 <experiment_name> [hydra_overrides...]"
    exit 1
fi
experiment=$1
shift

send_telegram "Experiment: $experiment (Cloud run @ $CLOUD_PREFIX)"

echo "--------------------------------------------------"
send_telegram "📌 Pretrained checkpoint [$(date "+%Y-%m-%d %H:%M:%S")]"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 1: reben_7k"
python -m train.finetune "$@" experiment_name="$experiment" paths=google paths.cloud_prefix="$CLOUD_PREFIX" num_workers=2 prefetch_factor=2 segmentation_ds=reben_7k &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 2: sen1floods11"
python -m train.finetune "$@" experiment_name="$experiment" paths=google paths.cloud_prefix="$CLOUD_PREFIX" num_workers=2 prefetch_factor=2 segmentation_ds=sen1floods11 &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune "$@" experiment_name="$experiment" paths=google paths.cloud_prefix="$CLOUD_PREFIX" num_workers=2 prefetch_factor=2 segmentation_ds=cashew_plantation &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 4: c2seg"
python -m train.finetune "$@" experiment_name="$experiment" paths=google paths.cloud_prefix="$CLOUD_PREFIX" num_workers=2 prefetch_factor=2 segmentation_ds=c2seg &
child_pid=$!
wait "$child_pid"
