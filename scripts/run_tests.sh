#!/usr/bin/env bash

echo "--------------------------------------------------"
echo "🎓 Testing 1.1: reben_7k"
python -m test.test "$@" segmentation_ds=reben_7k
echo "--------------------------------------------------"
echo "🎓 Testing 1.2: reben_7k (Out-Of-Distribution Set)"
python -m test.test "$@" segmentation_ds=reben_7k +ood_split=ood
echo "--------------------------------------------------"
echo "🎓 Testing 2.1: sen1floods11"
python -m test.test "$@" segmentation_ds=sen1floods11
echo "--------------------------------------------------"
echo "🎓 Testing 2.2: sen1floods11 (Out-Of-Distribution Set)"
python -m test.test "$@" segmentation_ds=sen1floods11 +ood_split=bolivia
echo "--------------------------------------------------"
echo "🎓 Testing 3: cashew_plantation"
python -m test.test "$@" segmentation_ds=cashew_plantation
echo "--------------------------------------------------"
echo "🎓 Testing 4: c2seg_wuhan"
python -m test.test "$@" segmentation_ds=c2seg dataset.city=wuhan
echo "--------------------------------------------------"
echo "🎓 Testing 5: burn_scars"
python -m test.test "$@" segmentation_ds=burn_scars

