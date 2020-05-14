#!/bin/bash
datasets="aircraft"
models="4conv"

for dataset in $datasets
do
    for model in $models
    do
        echo "dataset: ${dataset}, model: ${model}"
        python ./main.py --folder=/home/osilab7/hdd/ml_dataset \
                         --dataset=$dataset \
                         --model=$model \
                         --device=cuda:1 \
                         --batch-iter=300 \
                         --hidden-size=64 \
                         --num-ways=5 \
                         --num-shots=5 \
                         --extractor-step-size=0.5 \
                         --classifier-step-size=0.5 \
                         --meta-lr=1e-3 \
                         --download \
                         --save-name=5shot_4conv_both
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
                         --device=cuda:1 \
                         --batch-iter=300 \
                         --hidden-size=64 \
                         --num-ways=5 \
                         --num-shots=5 \
                         --extractor-step-size=0.5 \
                         --classifier-step-size=0.0 \
                         --meta-lr=1e-3 \
                         --save-name=5shot_4conv_extractor
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
#                          --device=cuda:1 \
#                          --batch-iter=300 \
#                          --hidden-size=64 \
#                          --num-ways=5 \
#                          --num-shots=5 \
#                          --extractor-step-size=0.0 \
#                          --classifier-step-size=0.5 \
#                          --meta-lr=1e-3 \
#                          --save-name=5shot_4conv_classifier
#     done
# done
    
echo "finished"
