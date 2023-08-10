#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: EPYC 7502 + GTX 1080 Ti
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

CUDA_VISIBLE_DEVICES=1 unbuffer python bab_verification_eran.py --device cuda --load eran_models/mnist_conv_big_diffai.pth --epsilon 0.3 --model mnist_conv_big --data MNIST_ERAN --batch_size 20 --iteration 20 --timeout 180 --branching_reduceop max --mode verified-acc --start 269 --end 270 2>&1 | tee reference_outputs/test1_safe.out
CUDA_VISIBLE_DEVICES=2 unbuffer python bab_verification_eran.py --device cuda --load eran_models/mnist_conv_big_diffai.pth --epsilon 0.3 --model mnist_conv_big --data MNIST_ERAN --batch_size 20 --iteration 20 --timeout 180 --branching_reduceop max --mode verified-acc --start 232 --end 233 2>&1 | tee reference_outputs/test2_safe.out
CUDA_VISIBLE_DEVICES=3 unbuffer python bab_verification_eran.py --device cuda --load eran_models/mnist_conv_big_diffai.pth --epsilon 0.3 --model mnist_conv_big --data MNIST_ERAN --batch_size 20 --iteration 20 --timeout 180 --branching_reduceop max --mode verified-acc --pgd_order skip --start 355 --end 356 2>&1 | tee reference_outputs/test3_unsafe.out
CUDA_VISIBLE_DEVICES=4 unbuffer python bab_verification_eran.py --device cuda --load eran_models/mnist_conv_big_diffai.pth --epsilon 0.3 --model mnist_conv_big --data MNIST_ERAN --batch_size 20 --iteration 20 --timeout 180 --branching_reduceop max --mode verified-acc --pgd_order skip --start 133 --end 134 2>&1 | tee reference_outputs/test4_unsafe.out