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

python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 26 --end 27 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 73 --end 74 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 65 --end 66 2>&1 | tee reference_outputs/test3_unsafe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 118 --end 119 2>&1 | tee reference_outputs/test4_safe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 96 --end 97 2>&1 | tee reference_outputs/test5_unsafe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 135 --end 136 2>&1 | tee reference_outputs/test6_safe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 143 --end 144 2>&1 | tee reference_outputs/test7_unsafe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 180 --end 181 2>&1 | tee reference_outputs/test8_safe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 181 --end 182 2>&1 | tee reference_outputs/test9_safe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 182 --end 183 2>&1 | tee reference_outputs/test10_unsafe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 183 --end 184 2>&1 | tee reference_outputs/test11_unsafe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 184 --end 185 2>&1 | tee reference_outputs/test12_safe.out
python bab_verification_input_split.py --share_slopes --no_solve_slope --data ACASXU  --load ../../../../vnncomp2021/benchmarks/acasxu --csv_name acasxu_instances.csv --start 185 --end 186 2>&1 | tee reference_outputs/test13_safe.out
