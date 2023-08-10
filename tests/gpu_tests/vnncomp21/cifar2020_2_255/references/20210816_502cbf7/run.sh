#!/bin/bash

# Testing hardware: EPYC 7502 + GTX 1080 Ti
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

python bab_verification_general.py --data CIFAR --batch_size 200 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../vnncomp2021/benchmarks/cifar2020 --csv_name cifar2020_instances.csv --start 27 --end 28 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_general.py --data CIFAR --batch_size 200 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../vnncomp2021/benchmarks/cifar2020 --csv_name cifar2020_instances.csv --start 44 --end 45 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_general.py --data CIFAR --batch_size 200 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../vnncomp2021/benchmarks/cifar2020 --csv_name cifar2020_instances.csv --start 5 --end 6 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_general.py --data CIFAR --batch_size 200 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../vnncomp2021/benchmarks/cifar2020 --csv_name cifar2020_instances.csv --pgd_order skip --start 0 --end 1 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_general.py --data CIFAR --batch_size 200 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --load ../../../vnncomp2021/benchmarks/cifar2020 --csv_name cifar2020_instances.csv --pgd_order skip --start 70 --end 71 2>&1 | tee reference_outputs/test5_unsafe.out
