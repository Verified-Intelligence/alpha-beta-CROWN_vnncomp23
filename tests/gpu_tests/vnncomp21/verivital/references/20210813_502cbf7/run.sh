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

ARGUMENTS="--complete_verifier mip"

python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/verivital --csv_name verivital_instances.csv --data MNIST --start 27 --end 28 $ARGUMENTS --pgd_order after 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/verivital --csv_name verivital_instances.csv --data MNIST --start 32 --end 33 $ARGUMENTS --pgd_order after 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/verivital --csv_name verivital_instances.csv --data MNIST --start 4 --end 5 $ARGUMENTS --pgd_order skip 2>&1 | tee reference_outputs/test3_unsafe.out
python bab_verification_general.py  --load ../../vnncomp2021/benchmarks/verivital --csv_name verivital_instances.csv --data MNIST --start 54 --end 55 $ARGUMENTS --pgd_order skip 2>&1 | tee reference_outputs/test4_unsafe.out

cd ..
rm -rf reference_outputs
mv src/reference_outputs .