#!/usr/bin/env bash
#SBATCH --job-name=rs_mae_train

#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=gpu_short
#SBATCH --time=1-00:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=aloui@campus.tu-berlin.de
#SBATCH --output=/scratch/jacer/outputs/slurm/job-%j.slurm.out
#SBATCH --chdir=/home/users/j/jacer/projects/rs_mae/


# import SECRETS_PATH, get_value, send_telegram
source /home/users/j/jacer/projects/rs_mae/scripts/functions.sh

CLEANUP_DONE=false
on_exit() {
    exit_code=$?
    # Check if this function has already run
    if [ "$CLEANUP_DONE" = true ]; then return; fi
    CLEANUP_DONE=true
    # Kill background python process if it's still running
    if [ -n "$child_pid" ]; then kill "$child_pid" 2>/dev/null || true; fi

    set +e  # Disable strict exit for the cleanup
    
    if [ "$exit_code" -eq 0 ]; then send_telegram "✅ Finished training successfully at $(date "+%Y-%m-%d %H:%M:%S")"
    else send_telegram "❌ Training failed (exit code $exit_code) on $(hostname) at $(date "+%Y-%m-%d %H:%M:%S")"
    fi
    #rsync -av /scratch/jacer/outputs/ /home/users/j/jacer/outputs/
    #send_telegram "📁 Finished copying outputs to home directory"
    exit $exit_code
}
trap on_exit EXIT TERM INT

cat <<'MOCK_TRAIN'
mkdir -p "/scratch/jacer/outputs/slurm/"
mkdir -p "/scratch/jacer/outputs/training_out/" # Specified in configs/paths/cluster.yaml as wandb_outdir

set -e # Strict processing instructions

module load nvidia/cuda/12.1
module load nvidia/cudnn/9.0.0

export MAMBA_ROOT_PREFIX=/home/users/j/jacer/programs/micromamba
eval "$(/home/users/j/jacer/programs/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /beegfs/scratch/jacer/envs/rs_mae

export WANDB_API_KEY=$(get_value "$SECRETS_PATH" "WANDB_API_KEY")
export WANDB_MODE=${WANDB_MODE:-online}

nvidia-smi || echo -e "\e[31mNo NVIDIA GPU available on this node\e[0m"
lscpu | grep -E "Model name|CPU\(s\):"
free -h
echo -e "• Node: $(hostname) • SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST\n• SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE\n• SLURM_GTIDS: $SLURM_GTIDS"
echo "Using python: $(which python)"
MOCK_TRAIN

echo "--------------------------------------------------"
send_telegram "🚀 Pretraining [$(date "+%Y-%m-%d %H:%M:%S")]"
python -m train.pretrain "$@" paths=local batch_size=64 num_workers=4 patch_size=32 max_epochs=1 img_size=96 experiment=feat_guided target.gaussian_window=3 &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 1: reben_7k"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 segmentation_ds=reben_7k &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 2: sen1floods11"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 img_size=192 segmentation_ds=sen1floods11 &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 segmentation_ds=cashew_plantation &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 4: c2seg"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 img_size=192 segmentation_ds=c2seg dataset.crop_size=192 &
child_pid=$!
wait "$child_pid"
echo "--------------------------------------------------"
send_telegram "🚀 Finetuning 5: burn_scars"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 img_size=192 segmentation_ds=burn_scars &
child_pid=$!
wait "$child_pid"
