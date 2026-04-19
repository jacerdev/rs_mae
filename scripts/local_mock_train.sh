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
#source /home/users/j/jacer/projects/rs_mae/scripts/functions.sh

#trap on_exit EXIT TERM INT

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> [hydra_overrides...]"
    exit 1
fi
experiment=$1
shift

echo "Experiment: $experiment"

echo "--------------------------------------------------"
echo "🚀 Pretraining [$(date "+%Y-%m-%d %H:%M:%S")]"
#python -m train.pretrain "$@" paths=local batch_size=64 num_workers=4 patch_size=32 max_epochs=1 img_size=96 experiment="$experiment" target.gaussian_window=3 & # feat_guided
python -m train.pretrain "$@" paths=local batch_size=64 num_workers=4 patch_size=32 max_epochs=1 img_size=96 experiment="$experiment"
echo "--------------------------------------------------"
echo "🚀 Finetuning 1: reben_7k"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 segmentation_ds=reben_7k experiment_name="$experiment"
echo "--------------------------------------------------"
echo "🚀 Finetuning 2: sen1floods11"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 img_size=192 segmentation_ds=sen1floods11 experiment_name="$experiment"
echo "--------------------------------------------------"
echo "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 segmentation_ds=cashew_plantation experiment_name="$experiment"
echo "--------------------------------------------------"
echo "🚀 Finetuning 4: c2seg"
python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 img_size=192 segmentation_ds=c2seg dataset.crop_size=192 experiment_name="$experiment"
#echo "--------------------------------------------------"
#echo "🚀 Finetuning 5: burn_scars"
#python -m train.finetune "$@" paths=local batch_size=16 num_workers=4 patch_size=32 max_epochs=1 img_size=192 segmentation_ds=burn_scars experiment_name="$experiment"
