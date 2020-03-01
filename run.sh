#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:1 \
                 --download \
                 --task-embedding-method=gcn \
                 --edge-generation-method=manual \
                 --save-name=test
echo "finished"
