#!/bin/bash

# Testing hardware: EPYC 7502 + GTX 1080 Ti
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

python bab_verification_eran.py --load eran_models/cifar_conv_big_pgd.pth --model cifar_conv_big --data CIFAR_ERAN --batch_size 64 --iteration 20 --timeout 180 --mode verified-acc --start 799 --end 800 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_eran.py --load eran_models/cifar_conv_big_pgd.pth --model cifar_conv_big --data CIFAR_ERAN --batch_size 64 --iteration 20 --timeout 180 --mode verified-acc --start 580 --end 581 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_eran.py --load eran_models/cifar_conv_big_pgd.pth --model cifar_conv_big --data CIFAR_ERAN --batch_size 64 --iteration 20 --timeout 180 --mode verified-acc --start 187 --end 188 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_eran.py --load eran_models/cifar_conv_big_pgd.pth --model cifar_conv_big --data CIFAR_ERAN --batch_size 64 --iteration 20 --timeout 180 --mode verified-acc --pgd_order skip --start 444 --end 445 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_eran.py --load eran_models/cifar_conv_big_pgd.pth --model cifar_conv_big --data CIFAR_ERAN --batch_size 64 --iteration 20 --timeout 180 --mode verified-acc --pgd_order skip --start 684 --end 685 2>&1 | tee reference_outputs/test5_unsafe.out
