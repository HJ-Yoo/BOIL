#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:1 \
		 --download \
                 --task-embedding-method=gcn \
                 --edge-generation-method=max_normalization \
		 --save-name=te_gcn_maxnorm_l1_output_normalization_relu \
		 --best-valid-accuracy-test
echo "finished"
