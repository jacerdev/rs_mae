#!/usr/bin/env bash
#SBATCH --job-name=rs_mae_train

#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=26
#SBATCH --partition=gpu_short
#SBATCH --time=0-14:00:00
#SBATCH --nodelist=gpu026

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

# --------------------------------------------------

if [ -z "$1" ]; then
    echo "Usage: sbatch $0 <experiment_name_1> [experiment_name_2...] [hydra_overrides...]"
    exit 1
fi

# Parse experiment names (stop at the first hydra override containing '=')
experiments=()
while [[ $# -gt 0 && "$1" != *=* ]]; do
    experiments+=("$1")
    shift
done
overrides="$@"

# Force temporary files to be created on the scratch drive to prevent cross-device link errors
export TMPDIR="/scratch/jacer/outputs/tmp"
mkdir -p $TMPDIR

# Loop over experiments with a 2-step jump for parallel pretraining
for ((i=0; i<${#experiments[@]}; i+=2)); do
    exp1=${experiments[$i]}
    exp2=${experiments[$i+1]}

    echo "=================================================="
    send_telegram "🚀 Pretraining: $exp1 & $exp2"
    echo "=================================================="

    python -m train.pretrain experiment="$exp1" $overrides &
    pid1=$!

    pid2=""
    if [ -n "$exp2" ]; then # run 2nd experiment in parallel if exists
        python -m train.pretrain experiment="$exp2" $overrides &
        pid2=$!
    fi

    wait $pid1 ${pid2:+$pid2}
done


datasets=("reben_7k" "c2seg_beijing" "cashew_plantation" "c2seg_augsburg" "mmhunan")

# Loop over datasets with a 2-step jump for parallel finetuning
for dataset in "${datasets[@]}"; do
    for ((i=0; i<${#experiments[@]}; i+=2)); do
        exp1=${experiments[$i]}
        exp2=${experiments[$i+1]}

        echo "=================================================="
        send_telegram "🚀 Finetuning $dataset: $exp1 & $exp2"
        echo "=================================================="

        python -m train.finetune \
            experiment_name="$exp1" segmentation_ds="$dataset" $overrides &
        pid1=$!

        pid2=""
        if [ ! -z "$exp2" ]; then
            python -m train.finetune \
                experiment_name="$exp2" segmentation_ds="$dataset" $overrides &
            pid2=$!
        fi

        wait $pid1 ${pid2:+$pid2}
    done
done

send_telegram "✅ All experiments and datasets finished."
