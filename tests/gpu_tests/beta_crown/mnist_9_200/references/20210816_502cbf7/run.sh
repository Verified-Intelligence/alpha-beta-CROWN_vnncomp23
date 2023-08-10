#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: EPYC 7502 + GTX 1080 Ti
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

# master can verify test1_unsafe within 298s
python bab_verification_eran.py --load eran_models/mnist_9_200_nat_old.pth --model mnist_9_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 500 --start 24 --end 25 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_eran.py --load eran_models/mnist_9_200_nat_old.pth --model mnist_9_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 500 --start 29 --end 30 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_eran.py --load eran_models/mnist_9_200_nat_old.pth --model mnist_9_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 500 --start 36 --end 37 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_eran.py --load eran_models/mnist_9_200_nat_old.pth --model mnist_9_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 500 --pgd_order skip --start 65 --end 66 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_eran.py --load eran_models/mnist_9_200_nat_old.pth --model mnist_9_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 500 --pgd_order skip --start 950 --end 951 2>&1 | tee reference_outputs/test5_unsafe.out