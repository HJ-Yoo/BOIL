#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:0 \
                 --download \
                 --task-embedding-method=gcn \
                 --edge-generation-method=max_normalization \
		 --meta-lr=1e-3 \
		 --init \
                 --save-name=init_test
echo "finished"
