#!/bin/bash

python ./main.py --folder=./data \
                 --dataset=cifar_fs \
                 --model=resnet \
                 --blocks-type=a \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.3 \
                 --classifier-step-size=0.3 \
                 --meta-lr=6e-4 \
                 --download \
                 --save-name=5shot_resnet_block_a_MAML
                 
python ./main.py --folder=./data \
                 --dataset=cifar_fs \
                 --model=resnet \
                 --blocks-type=a \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.3 \
                 --classifier-step-size=0.0 \
                 --meta-lr=6e-4 \
                 --download \
                 --save-name=5shot_resnet_block_a_BOIL
                 
python ./main.py --folder=./data \
                 --dataset=cifar_fs \
                 --model=resnet \
                 --blocks-type=b \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.3 \
                 --classifier-step-size=0.3 \
                 --meta-lr=6e-4 \
                 --download \
                 --save-name=5shot_resnet_block_b_MAML
                 
python ./main.py --folder=./data \
                 --dataset=cifar_fs \
                 --model=resnet \
                 --blocks-type=b \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --extractor-step-size=0.3 \
                 --classifier-step-size=0.0 \
                 --meta-lr=6e-4 \
                 --download \
                 --save-name=5shot_resnet_block_b_BOIL

echo "finished"
