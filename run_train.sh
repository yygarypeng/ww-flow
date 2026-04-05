# bin/bash

taskset -c 0-3 python train.py -w &> record &
