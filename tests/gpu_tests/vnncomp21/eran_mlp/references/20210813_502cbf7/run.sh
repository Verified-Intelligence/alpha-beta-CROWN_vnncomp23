#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: EPYC 7502 + GTX 1080 Ti
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd src
mkdir reference_outputs

# manually set mip_multi_proc = 16 at line 2378 in beta_CROWN_solver.py
# the last case is originally timeout during competition but now becomes safe on 1080ti
CUDA_VISIBLE_DEVICES=1 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 1 --end 2 --pgd_order before --complete_verifier bab-refine 2>&1 | tee reference_outputs/test1_safe.out
CUDA_VISIBLE_DEVICES=2 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 5 --end 6 --pgd_order before --complete_verifier bab-refine 2>&1 | tee reference_outputs/test2_safe.out
CUDA_VISIBLE_DEVICES=3 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 24 --end 25 --pgd_order before --complete_verifier bab-refine 2>&1 | tee reference_outputs/test3_safe.out
CUDA_VISIBLE_DEVICES=4 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 17 --end 18 --pgd_order before --complete_verifier bab-refine 2>&1 | tee reference_outputs/test4_timeout.out
CUDA_VISIBLE_DEVICES=5 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/eran --csv_name eran_instances.csv --data MNIST --start 33 --end 34 --pgd_order before --complete_verifier bab-refine 2>&1 | tee reference_outputs/test5_safe.out
