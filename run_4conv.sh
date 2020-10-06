#!/bin/bash

python ./main.py --folder=./data \
                 --dataset=cifar_fs \
                 --model=4conv \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --batch-iter=50 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.5 \
                 --meta-lr=1e-3 \
                 --download \
                 --save-name=5shot_4conv_MAML
                 
python ./main.py --folder=./data \
                 --dataset=cifar_fs \
                 --model=4conv \
                 --device=cuda:0 \
                 --num-ways=5 \
                 --num-shots=5 \
                 --batch-iter=50 \
                 --extractor-step-size=0.5 \
                 --classifier-step-size=0.0 \
                 --meta-lr=1e-3 \
                 --download \
                 --save-name=5shot_4conv_BOIL

# python ./main.py --folder=./data \
#                  --dataset=cifar_fs \
#                  --model=4conv \
#                  --device=cuda:0 \
#                  --num-ways=5 \
#                  --num-shots=5 \
#                  --extractor-step-size=0.5 \
#                  --classifier-step-size=0.5 \
#                  --meta-lr=1e-3 \
#                  --download \
#                  --ortho-init \
#                  --outer-fix \
#                  --save-name=5shot_4conv_MAML-fix
                 
# python ./main.py --folder=./data \
#                  --dataset=cifar_fs \
#                  --model=4conv \
#                  --device=cuda:0 \
#                  --num-ways=5 \
#                  --num-shots=5 \
#                  --extractor-step-size=0.5 \
#                  --classifier-step-size=0.0 \
#                  --meta-lr=1e-3 \
#                  --download \
#                  --ortho-init \
#                  --outer-fix \
#                  --save-name=5shot_4conv_BOIL-fix

echo "finished"
