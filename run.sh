#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:0 \
                 --download \
                 --task-embedding-method=gcn \
                 --edge-generation-method=manual \
                 --save-name=te_gcn_manual0.1_l1_concat
echo "finished"
