#!/usr/bin/env bash

# import SECRETS_PATH, get_value, send_telegram, on_exit
source scripts/functions.sh

# Re-define SECRETS_PATH for cloud environments
export SECRETS_PATH="cloud/personal.env"

trap on_exit EXIT TERM INT
mkdir -p 'cloud/outputs/logs/'
mkdir -p 'cloud/outputs/training_out/' # Specified in configs/paths/cloud.yaml as wandb_outdir

set -e # Strict processing instructions

# Environment activation
# Assumes everything is inside `cloud/envs/rs_mae`
# If your cloud provider already provisions the environment, comment this block out.
export MAMBA_EXE='cloud/micromamba/micromamba'
export MAMBA_ROOT_PREFIX='cloud/envs'
if [ -f "$MAMBA_EXE" ]; then
    eval "$("$MAMBA_EXE" shell hook --shell bash)"
    micromamba activate cloud/envs/rs_mae
else
    echo "Local Micromamba not found at $MAMBA_EXE, relying on system python"
fi

export WANDB_API_KEY=$(get_value "$SECRETS_PATH" "WANDB_API_KEY")
export WANDB_MODE=${WANDB_MODE:-online}

nvidia-smi || echo -e "\e[31mNo NVIDIA GPU available on this node\e[0m"
lscpu | grep -E "Model name|CPU\(s\):"
free -h
echo -e "• Node: $(hostname)"
echo "Using python: $(which python)"

if [ -z "$1" ]; then
    echo "Usage: bash $0 <experiment_name> [hydra_overrides...]"
    exit 1
fi
experiment=$1
shift

send_telegram "Experiment: $experiment (Cloud run)"

echo "--------------------------------------------------"
send_telegram "🚀 Pretraining [$(date "+%Y-%m-%d %H:%M:%S")]"
python -m train.pretrain "$@" experiment="$experiment" paths=cloud &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 1: reben_7k"
python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=reben_7k paths=cloud &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 2: sen1floods11"
python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=sen1floods11 paths=cloud &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=cashew_plantation paths=cloud &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 4: c2seg"
python -m train.finetune "$@" experiment_name="$experiment" segmentation_ds=c2seg paths=cloud &
child_pid=$!
wait "$child_pid"
