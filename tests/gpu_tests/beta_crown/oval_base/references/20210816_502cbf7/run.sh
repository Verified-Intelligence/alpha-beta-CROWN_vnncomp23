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

python bab_verification.py --device cuda --load ../data/cifar_base_kw.pth --model cifar_model --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 40 --mode complete --start 16 --end 17 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification.py --device cuda --load ../data/cifar_base_kw.pth --model cifar_model --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 40 --mode complete --start 17 --end 18 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification.py --device cuda --load ../data/cifar_base_kw.pth --model cifar_model --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 160 --mode complete --start 64 --end 65 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification.py --device cuda --load ../data/cifar_base_kw.pth --model cifar_model --data CIFAR --batch_size 1024 --branching_method fsb --branching_candidates 1 --timeout 240 --mode complete --start 66 --end 67 2>&1 | tee reference_outputs/test4_safe.out

