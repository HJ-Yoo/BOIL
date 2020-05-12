#!/bin/bash
datasets="cifar_fs aircraft vgg_flower"
models="resnet"
types="a b"

for type in $types
do
    for dataset in $datasets
    do
        for model in $models
        do
            echo "dataset: ${dataset}, model: ${model}"
            python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
                             --dataset=$dataset \
                             --model=$model \
                             --device=cuda:1 \
                             --batch-iter=200 \
                             --num-ways=5 \
                             --num-shots=5 \
                             --download \
                             --extractor-step-size=0.3 \
                             --classifier-step-size=0.0 \
                             --meta-lr=6e-4 \
                             --blocks-type=$type \
                             --save-name=5shot_resnet_block_${type}_extractor
        done
    done
done

for type in $types
do
    for dataset in $datasets
    do
        for model in $models
        do
            echo "dataset: ${dataset}, model: ${model}"
            python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
                             --dataset=$dataset \
                             --model=$model \
                             --device=cuda:1 \
                             --batch-iter=200 \
                             --num-ways=5 \
                             --num-shots=5 \
                             --download \
                             --extractor-step-size=0.3 \
                             --classifier-step-size=0.3 \
                             --meta-lr=6e-4 \
                             --blocks-type=$type \
                             --save-name=5shot_resnet_block_${type}_both
        done
    done
done

# for dataset in $datasets
# do
#     for model in $models
#     do
#         echo "dataset: ${dataset}, model: ${model}"
#         python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
#                          --dataset=$dataset \
#                          --model=$model \
#                          --device=cuda:0 \
#                          --batch-iter=300 \
#                          --num-ways=5 \
#                          --num-shots=5 \
#                          --download \
#                          --extractor-step-size=0.0 \
#                          --classifier-step-size=0.5 \
#                          --meta-lr=1e-3 \
#                          --blocks-type=a \
#                          --save-name=5shot_resnet_block_a_classifier
#     done
# done
    
echo "finished"
