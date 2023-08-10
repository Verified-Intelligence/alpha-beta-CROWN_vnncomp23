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

# manually set mip_multi_proc = 16 at line 1789 in beta_CROWN_solver.py
CUDA_VISIBLE_DEVICES=1 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/mnistfc --csv_name mnistfc_instances.csv --data MNIST --start 5 --end 6 --pgd_order before --complete_verifier mip --branching_candidate 5 --branching_reduceop max --lr_beta 0.003 2>&1 | tee reference_outputs/test1_safe.out
CUDA_VISIBLE_DEVICES=2 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/mnistfc --csv_name mnistfc_instances.csv --data MNIST --start 21 --end 22 --pgd_order before --complete_verifier mip --branching_candidate 5 --branching_reduceop max --lr_beta 0.003 2>&1 | tee reference_outputs/test2_safe.out
CUDA_VISIBLE_DEVICES=3 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/mnistfc --csv_name mnistfc_instances.csv --data MNIST --start 25 --end 26 --pgd_order before --complete_verifier mip --branching_candidate 5 --branching_reduceop max --lr_beta 0.003 2>&1 | tee reference_outputs/test3_safe.out
CUDA_VISIBLE_DEVICES=4 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/mnistfc --csv_name mnistfc_instances.csv --data MNIST --start 7 --end 8 --pgd_order skip --complete_verifier mip --branching_candidate 5 --branching_reduceop max --lr_beta 0.003 2>&1 | tee reference_outputs/test4_unsafe.out
CUDA_VISIBLE_DEVICES=5 unbuffer python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/mnistfc --csv_name mnistfc_instances.csv --data MNIST --start 15 --end 16 --pgd_order skip --complete_verifier mip --branching_candidate 5 --branching_reduceop max --lr_beta 0.003 2>&1 | tee reference_outputs/test5_unsafe.out
