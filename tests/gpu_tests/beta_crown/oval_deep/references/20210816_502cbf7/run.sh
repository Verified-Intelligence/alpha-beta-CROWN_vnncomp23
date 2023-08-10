#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: AMD Ryzen 9 5950X + GTX 3090
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

mkdir reference_outputs

python bab_verification.py --device cuda --load ../data/cifar_deep_kw.pth --model cifar_model_deep --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 12 --mode complete --start 3 --end 4 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification.py --device cuda --load ../data/cifar_deep_kw.pth --model cifar_model_deep --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 14 --mode complete --start 10 --end 11 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification.py --device cuda --load ../data/cifar_deep_kw.pth --model cifar_model_deep --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 18 --mode complete --start 20 --end 21 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification.py --device cuda --load ../data/cifar_deep_kw.pth --model cifar_model_deep --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 12 --mode complete --start 91 --end 92 2>&1 | tee reference_outputs/test4_safe.out

