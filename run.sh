#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:0\
                 --download \
                 --adaptive-lr \
                 --save-name=adaptive_lr_inner
                 
python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:0\
                 --download \
                 --graph-regularizer \
                 --graph-beta=1e-2 \
                 --save-name=new_gr1e-2_outer
                 
python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:0\
                 --download \
                 --graph-regularizer \
                 --graph-beta=1e-2 \
                 --adaptive-lr \
                 --save-name=adaptive_lr_inner_new_gr1e-2_outer
                 
echo "finished"
