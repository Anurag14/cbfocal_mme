#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python main_stage2.py --method $1 --dataset multi --source $2 --target $3 --net $4 #--save_check
