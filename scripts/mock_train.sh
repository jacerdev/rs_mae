#!/usr/bin/env bash


if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> [hydra_overrides...]"
    exit 1
fi
experiment=$1
shift

echo "Experiment: $experiment"

echo "--------------------------------------------------"
echo "🚀 Pretraining [$(date "+%Y-%m-%d %H:%M:%S")]"
#python -m train.pretrain paths=local batch_size=64 num_workers=4 patch_size=32 max_epochs=1 img_size=96 experiment="$experiment" target.gaussian_window=3 & # feat_guided
python -m train.pretrain paths=local +is_dummy=true batch_size=64 log_every_n_steps=20 num_workers=4 patch_size=32 max_epochs=1 precision="16-mixed" img_size=96 experiment="$experiment" "$@"
echo "--------------------------------------------------"
echo "🚀 Finetuning 1: reben_7k"
python -m train.finetune paths=local +is_dummy=true batch_size=16 num_workers=4 patch_size=32 max_epochs=1 precision="16-mixed" segmentation_ds=reben_7k experiment_name="$experiment" "$@"
echo "--------------------------------------------------"
echo "🚀 Finetuning 2: mmhunan"
python -m train.finetune paths=local +is_dummy=true batch_size=16 num_workers=4 patch_size=32 max_epochs=1 precision="16-mixed" segmentation_ds=mmhunan experiment_name="$experiment" "$@"
echo "--------------------------------------------------"
echo "🚀 Finetuning 3: cashew_plantation"
python -m train.finetune paths=local +is_dummy=true batch_size=16 num_workers=4 patch_size=32 max_epochs=1 precision="16-mixed" segmentation_ds=cashew_plantation experiment_name="$experiment" "$@"
echo "--------------------------------------------------"
echo "🚀 Finetuning 4: c2seg_ab"
python -m train.finetune paths=local +is_dummy=true batch_size=16 num_workers=4 patch_size=32 max_epochs=1 precision="16-mixed" segmentation_ds=c2seg_ab experiment_name="$experiment" "$@"
echo "--------------------------------------------------"
echo "🚀 Finetuning 4: c2seg_bw"
python -m train.finetune paths=local +is_dummy=true batch_size=16 num_workers=4 patch_size=32 max_epochs=1 precision="16-mixed" img_size=256 dataset.crop_size=256 segmentation_ds=c2seg_bw experiment_name="$experiment" "$@"
