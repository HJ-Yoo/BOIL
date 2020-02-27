#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:0 \
                 --download \
		         --save-name=graph_regularizer 
echo "finished"
