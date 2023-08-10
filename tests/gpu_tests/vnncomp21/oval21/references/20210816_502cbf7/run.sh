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

python bab_verification_general.py --data CIFAR --batch_size 2000 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../../vnncomp2021/benchmarks/oval21 --csv_name oval21_instances.csv --pgd_order skip --start 3 --end 4 2>&1 | tee reference_outputs/test1_unsafe.out
python bab_verification_general.py --data CIFAR --batch_size 2000 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../../vnncomp2021/benchmarks/oval21 --csv_name oval21_instances.csv --pgd_order skip --start 7 --end 8 2>&1 | tee reference_outputs/test2_unsafe.out
python bab_verification_general.py --data CIFAR --batch_size 2000 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../../vnncomp2021/benchmarks/oval21 --csv_name oval21_instances.csv --pgd_order skip --start 16 --end 17 2>&1 | tee reference_outputs/test3_unsafe.out
python bab_verification_general.py --data CIFAR --batch_size 2000 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../../vnncomp2021/benchmarks/oval21 --csv_name oval21_instances.csv --start 2 --end 3 2>&1 | tee reference_outputs/test4_safe.out
python bab_verification_general.py --data CIFAR --batch_size 2000 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../../vnncomp2021/benchmarks/oval21 --csv_name oval21_instances.csv --start 13 --end 14 2>&1 | tee reference_outputs/test5_safe.out
python bab_verification_general.py --data CIFAR --batch_size 2000 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../../vnncomp2021/benchmarks/oval21 --csv_name oval21_instances.csv --start 23 --end 24 2>&1 | tee reference_outputs/test6_safe.out
