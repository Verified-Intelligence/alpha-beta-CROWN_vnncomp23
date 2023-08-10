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
python bab_verification_eran.py --load eran_models/mnist_6_200_nat_old.pth --model mnist_6_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 360 --start 15 --end 16 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_eran.py --load eran_models/mnist_6_200_nat_old.pth --model mnist_6_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 300 --start 43 --end 44 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_eran.py --load eran_models/mnist_6_200_nat_old.pth --model mnist_6_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 300 --start 5 --end 6 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_eran.py --load eran_models/mnist_6_200_nat_old.pth --model mnist_6_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 300 --pgd_order skip --start 465 --end 466 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_eran.py --load eran_models/mnist_6_200_nat_old.pth --model mnist_6_200 --epsilon 0.015 --data MNIST_ERAN_UN --mode verified-acc --complete_verifier bab-refine --batch_size 500 --mip_multi_proc 16 --timeout 300 --pgd_order skip --start 199 --end 200 2>&1 | tee reference_outputs/test5_unsafe.out