#!/bin/bash
datasets="miniimagenet"
models="resnet"

for dataset in $datasets
do
    for model in $models
    do
        echo "dataset: ${dataset}, model: ${model}"
        python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
                         --dataset=$dataset \
                         --model=$model \
                         --device=cuda:0 \
                         --batch-iter=300 \
                         --num-ways=5 \
                         --num-shots=5 \
                         --download \
                         --extractor-step-size=0.5 \
                         --classifier-step-size=0.5 \
                         --meta-lr=1e-3 \
                         --blocks-type=a \
                         --save-name=5shot_resnet_block_a_both
    done
done

for dataset in $datasets
do
    for model in $models
    do
        echo "dataset: ${dataset}, model: ${model}"
        python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
                         --dataset=$dataset \
                         --model=$model \
                         --device=cuda:0 \
                         --batch-iter=300 \
                         --num-ways=5 \
                         --num-shots=5 \
                         --download \
                         --extractor-step-size=0.5 \
                         --classifier-step-size=0.0 \
                         --meta-lr=1e-3 \
                         --blocks-type=a \
                         --save-name=5shot_resnet_block_a_extractor
    done
done

for dataset in $datasets
do
    for model in $models
    do
        echo "dataset: ${dataset}, model: ${model}"
        python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
                         --dataset=$dataset \
                         --model=$model \
                         --device=cuda:0 \
                         --batch-iter=300 \
                         --num-ways=5 \
                         --num-shots=5 \
                         --download \
                         --extractor-step-size=0.0 \
                         --classifier-step-size=0.5 \
                         --meta-lr=1e-3 \
                         --blocks-type=a \
                         --save-name=5shot_resnet_block_a_classifier
    done
done
    
echo "finished"
