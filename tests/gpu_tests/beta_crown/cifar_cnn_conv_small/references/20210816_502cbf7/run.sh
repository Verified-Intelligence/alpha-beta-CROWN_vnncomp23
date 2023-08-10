#!/bin/bash

# Testing hardware: EPYC 7502 + GTX 1080 Ti
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

python bab_verification_eran.py --load eran_models/cifar_conv_small_pgd.pth --model cifar_conv_small --data CIFAR_ERAN --batch_size 2048 --iteration 20 --timeout 120 --mode verified-acc --start 373 --end 374 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_eran.py --load eran_models/cifar_conv_small_pgd.pth --model cifar_conv_small --data CIFAR_ERAN --batch_size 2048 --iteration 20 --timeout 120 --mode verified-acc --start 816 --end 817 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_eran.py --load eran_models/cifar_conv_small_pgd.pth --model cifar_conv_small --data CIFAR_ERAN --batch_size 2048 --iteration 20 --timeout 120 --mode verified-acc --start 920 --end 921 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_eran.py --load eran_models/cifar_conv_small_pgd.pth --model cifar_conv_small --data CIFAR_ERAN --batch_size 2048 --iteration 20 --timeout 120 --mode verified-acc --pgd_order skip --start 761 --end 762 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_eran.py --load eran_models/cifar_conv_small_pgd.pth --model cifar_conv_small --data CIFAR_ERAN --batch_size 2048 --iteration 20 --timeout 120 --mode verified-acc --pgd_order skip --start 821 --end 822 2>&1 | tee reference_outputs/test5_unsafe.out
