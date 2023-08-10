#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: AMD Ryzen 9 5950X + GTX 3090
# Software version: Pytorch 1.8.2 LTS, Python 3.7

# double timeout, in this commit, the running time is longer
cd examples/vision/plnn

mkdir reference_outputs

python bab_verification.py --device cuda --load ../data/cifar_wide_kw.pth --model cifar_model_wide --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 20 --mode complete --start 0 --end 1 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification.py --device cuda --load ../data/cifar_wide_kw.pth --model cifar_model_wide --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 48 --mode complete --start 34 --end 35 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification.py --device cuda --load ../data/cifar_wide_kw.pth --model cifar_model_wide --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 300 --mode complete --start 89 --end 90 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification.py --device cuda --load ../data/cifar_wide_kw.pth --model cifar_model_wide --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 24 --mode complete --start 94 --end 95 2>&1 | tee reference_outputs/test4_safe.out

