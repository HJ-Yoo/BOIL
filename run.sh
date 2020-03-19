#!/bin/bash

python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:1 \
                 --num-shots=5 \
                 --download \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.5 \
                 --save-name=both_inner
                 
python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:1 \
                 --num-shots=5 \
                 --download \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.0 \
                 --save-name=extractor_inner
                 
python ./main.py --folder=./dataset \
                 --dataset=miniimagenet \
                 --device=cuda:1 \
                 --num-shots=5 \
                 --download \
                 --extractor-step-size=0.0 \
                 --classifier-step-size=0.5 \
                 --save-name=classifier_inner

echo "finished"
