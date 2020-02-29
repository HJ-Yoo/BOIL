#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:0 \
                 --download \
		         --save-name=GR_beta_1.1e-5_gamma_6.0 \
                 --graph-gamma=6.0 \
                 --graph-beta=1.1e-5 
echo "finished"
