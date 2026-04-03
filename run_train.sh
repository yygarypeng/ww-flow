# bin/bash

taskset -c 4-7 python train.py -w &> record &
