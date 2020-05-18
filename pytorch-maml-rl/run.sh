#!/bin/bash
models="halfcheetah-vel 2d-navigation halfcheetah-dir"

for model in $models
do
    echo "maml_model : ${model}"
    python train.py --config configs/maml/${model}.yaml \
                --output-folder ./output/maml-${model} \
                --extractor_inner_step_size 0.1 \
                --classifier_inner_step_size 0.1 \
                --seed 2020 \
                --num-workers 16
done     


echo "finished"