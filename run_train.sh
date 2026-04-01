# bin/bash

# flow -> use 0-3 (4-7 for ww-flow) 
taskset -c 0-3 python train.py -w &> record &
