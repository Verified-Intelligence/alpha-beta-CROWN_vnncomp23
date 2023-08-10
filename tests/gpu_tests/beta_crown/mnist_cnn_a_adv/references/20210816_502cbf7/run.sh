#!/bin/bash

# Commands to reproduce the results on this old commit version. This file is
# for record only; it will not be run by automatic test. But you should make
# sure the command listed here are accurate so if something went wrong we can
# easily reproduce.

# Record key hardware and software configurations when the reference is generated.
# Testing hardware: EPYC 7502 + GTX 1080 Ti
# Software version: Pytorch 1.8.2 LTS, Python 3.7

cd examples/vision/plnn

python bab_verification_eran.py --load sdp_models/mnist_0.3_cnn_a_adv.model --model mnist_cnn_4layer --data MNIST_SDP --batch_size 600 --iteration 20 --timeout 120 --branching_reduceop max --mode verified-acc --start 2 --end 3 2>&1 | tee reference_outputs/test1_safe.out
python bab_verification_eran.py --load sdp_models/mnist_0.3_cnn_a_adv.model --model mnist_cnn_4layer --data MNIST_SDP --batch_size 600 --iteration 20 --timeout 120 --branching_reduceop max --mode verified-acc --start 69 --end 70 2>&1 | tee reference_outputs/test2_safe.out
python bab_verification_eran.py --load sdp_models/mnist_0.3_cnn_a_adv.model --model mnist_cnn_4layer --data MNIST_SDP --batch_size 600 --iteration 20 --timeout 120 --branching_reduceop max --mode verified-acc --start 91 --end 92 2>&1 | tee reference_outputs/test3_safe.out
python bab_verification_eran.py --load sdp_models/mnist_0.3_cnn_a_adv.model --model mnist_cnn_4layer --data MNIST_SDP --batch_size 600 --iteration 20 --timeout 120 --branching_reduceop max --mode verified-acc --pgd_order skip --start 30 --end 31 2>&1 | tee reference_outputs/test4_unsafe.out
python bab_verification_eran.py --load sdp_models/mnist_0.3_cnn_a_adv.model --model mnist_cnn_4layer --data MNIST_SDP --batch_size 600 --iteration 20 --timeout 120 --branching_reduceop max --mode verified-acc --pgd_order skip --start 112 --end 113 2>&1 | tee reference_outputs/test5_unsafe.out
