#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:0 \
                 --download \
		 --model-name=gcn_test 
echo "finished"
