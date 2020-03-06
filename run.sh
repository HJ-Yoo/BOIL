#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:1 \
                 --download \
		 --step-size=0.01 \
                 --save-name=step_size0.01
echo "finished"
