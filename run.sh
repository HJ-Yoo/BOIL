#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:1 \
                 --download \
                 --save-name=maml
echo "finished"
