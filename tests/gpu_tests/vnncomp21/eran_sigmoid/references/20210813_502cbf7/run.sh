#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: sonya.cs.ucla.edu
# Software version: Pytorch 1.8.1 LTS, Python 3.9.5

cd src
mkdir reference_outputs

python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 37 --end 38 --pgd_order before --complete_verifier skip --loss_reduction_func=min 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 38 --end 39 --pgd_order before --complete_verifier skip --loss_reduction_func=min 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 59 --end 60 --pgd_order before --complete_verifier skip --loss_reduction_func=min 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 54 --end 55 --pgd_order before --complete_verifier skip --loss_reduction_func=min 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 54 --end 55 --pgd_order skip --complete_verifier skip --loss_reduction_func=min 2>&1 | tee reference_outputs/test4_unsafe_no_PGD.out
