#!/bin/sh

echo 'start training'
python train.py --proxy True
python train.py --local-attn True -r sysu_agw_p4_n8_lr_0.1_seed_0_local_attn_True_proxy_False_best.t 
