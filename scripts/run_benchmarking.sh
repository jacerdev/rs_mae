#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> [hydra_overrides...]"
    exit 1
fi
experiment=$1
shift

echo "Experiment: $experiment"

echo "--------------------------------------------------"
echo "🎓 Testing 1: reben_7k"
python -m test.benchmark experiment_name=$experiment segmentation_ds=reben_7k paths=local paths.reben_dir="/mnt/ssd/datasets/reben/reben_7k/" "$@"
echo "--------------------------------------------------"
echo "🎓 Testing 2: sen1floods11"
python -m test.benchmark experiment_name=$experiment segmentation_ds=sen1floods11 paths=local "$@"
echo "--------------------------------------------------"
echo "🎓 Testing 3: cashew_plantation"
python -m test.benchmark experiment_name=$experiment segmentation_ds=cashew_plantation paths=local "$@"
echo "--------------------------------------------------"
echo "🎓 Testing 4: c2seg_wuhan"
python -m test.benchmark experiment_name=$experiment segmentation_ds=c2seg dataset.city=wuhan paths=local "$@"

