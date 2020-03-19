#!/bin/bash
datasets="miniimagenet tieredimagenet cifar_fs fc100"
models="smallconv largeconv"

for dataset in $datasets
do
    for model in $models
    do
        echo "dataset: ${dataset}, model: ${model}"
        python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
                         --dataset=$dataset \
                         --model=$model \
                         --device=cuda:1 \
                         --hidden-size=64 \
                         --num-shots=5 \
                         --download \
                         --extractor-step-size=0.5 \
                         --classifier-step-size=0.5 \
                         --save-name=5shot_${model}_both
    done
done
echo "finished"