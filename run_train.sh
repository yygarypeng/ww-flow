# !#/bin/bash

taskset -c 0-5 python train.py -w &> record &
