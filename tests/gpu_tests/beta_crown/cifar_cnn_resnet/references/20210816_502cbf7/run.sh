#!/bin/bash

# Testing hardware: Dual EPYC 7282 + RTX A6000
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

# Must change the RESNET dataloader in plnn/utils.py, load 10000 instead of 1000 examples.

# Requires > 30GB GPU memory.

python bab_verification_eran.py --load ../data/cifar_resnet_8px.pth --model model_resnet --data RESNET --mode verified-acc --iteration 20 --batch_size 8 --timeout 300 --mode verified-acc --start 4854 --end 4855 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_eran.py --load ../data/cifar_resnet_8px.pth --model model_resnet --data RESNET --mode verified-acc --iteration 20 --batch_size 8 --timeout 300 --mode verified-acc --start 9134 --end 9135 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_eran.py --load ../data/cifar_resnet_8px.pth --model model_resnet --data RESNET --mode verified-acc --iteration 20 --batch_size 8 --timeout 300 --mode verified-acc --start 3529 --end 3530 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_eran.py --load ../data/cifar_resnet_8px.pth --model model_resnet --data RESNET --mode verified-acc --iteration 20 --batch_size 8 --timeout 300 --mode verified-acc --pgd_order skip --start 240 --end 241 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_eran.py --load ../data/cifar_resnet_8px.pth --model model_resnet --data RESNET --mode verified-acc --iteration 20 --batch_size 8 --timeout 300 --mode verified-acc --pgd_order skip --start 193 --end 194 2>&1 | tee reference_outputs/test5_unsafe.out
