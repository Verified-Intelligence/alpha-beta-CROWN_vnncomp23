#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: sonya.cs.ucla.edu
# Software version: Pytorch 1.8.1 LTS, Python 3.7.11

cd src
rm -rf reference_outputs
mkdir reference_outputs

ARGUMENTS="--batch_size 2000 --lr_beta 0.01 --pgd_order skip --branching_reduceop max"

python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/cifar10_resnet --csv_name cifar10_resnet_instances.csv --data CIFAR --start 0 --end 1 $ARGUMENTS --pgd_order after 2>&1 | tee reference_outputs/test1.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/cifar10_resnet --csv_name cifar10_resnet_instances.csv --data CIFAR --start 62 --end 63 $ARGUMENTS --pgd_order after 2>&1 | tee reference_outputs/test2.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/cifar10_resnet --csv_name cifar10_resnet_instances.csv --data CIFAR --start 2 --end 3 $ARGUMENTS --pgd_order skip 2>&1 | tee reference_outputs/test3.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/cifar10_resnet --csv_name cifar10_resnet_instances.csv --data CIFAR --start 61 --end 62 $ARGUMENTS --pgd_order skip 2>&1 | tee reference_outputs/test4.out

cd ..
rm -rf reference_outputs
mv src/reference_outputs .