#!/usr/bin/env bash
#SBATCH --job-name=rs_mae_train

#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150G
#SBATCH --cpus-per-gpu=26
#SBATCH --partition=gpu_short
#SBATCH --time=0-14:00:00
########SBATCH --nodelist=gpu026

#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com
#SBATCH --output=/scratch/%u/outputs/slurm/job-%j.slurm.out

if [ -z "$1" ]; then
    echo "Usage: sbatch $0 <experiment_name_1> [experiment_name_2...] [hydra_overrides...]"
    exit 1
fi

cd "$HOME/projects/rs_mae/"

# import SECRETS_PATH, get_value, send_telegram, on_exit
source "$HOME/projects/rs_mae/scripts/functions.sh"

trap on_exit EXIT TERM INT
mkdir -p "/scratch/$USER/outputs/slurm/"
mkdir -p "/scratch/$USER/outputs/training_out/" # Specified in configs/paths/cluster.yaml as wandb_outdir

set -e # Strict processing instructions

# Environment activation
export MAMBA_EXE="$HOME/programs/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate /scratch/$USER/envs/rs_mae

# WandB
export WANDB_API_KEY=$(get_value "$SECRETS_PATH" "WANDB_API_KEY")
export WANDB_MODE=${WANDB_MODE:-online}

nvidia-smi || echo -e "\e[31mNo NVIDIA GPU available on this node\e[0m"
lscpu | grep -E "Model name|CPU\(s\):"
free -h
echo -e "• Node: $(hostname) • SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST\n• SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE\n• SLURM_GTIDS: $SLURM_GTIDS"
echo "Using python: $(which python)"

# --------------------------------------------------

# Parse experiment names (stop at the first hydra override containing '=')
experiments=()
while [[ $# -gt 0 && "$1" != *=* ]]; do
    experiments+=("$1")
    shift
done
overrides="$@"

# Force temporary files to be created on the scratch drive to prevent cross-device link errors
export TMPDIR="/scratch/$USER/outputs/tmp"
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


datasets=("c2seg_ab" "mmhunan" "cashew_plantation" "reben_7k" "c2seg_bw")

# Loop over datasets
for dataset in "${datasets[@]}"; do
    # Loop over experiments with a 2-step jump for parallel pretraining
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
