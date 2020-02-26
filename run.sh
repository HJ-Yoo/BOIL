#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:1 \
                 --download
echo "finished"
