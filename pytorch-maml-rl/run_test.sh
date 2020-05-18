#!/bin/bash
models="maml-halfcheetah-vel maml-2d-navigation maml-halfcheetah-dir ranil-halfcheetah-vel ranil-halfcheetah-dir ranil-2d-navigation maml-halfcheetah-vel2 ranil-halfcheetah-vel2"

for model in $models
do
    echo "model : ${model}"
    python test.py --config ./output/${model}/config.json \
                   --policy ./output/${model}/policy.th \
                   --output ./output/${model}/results.npz \
                   --meta-batch-size 20 \
                   --num-batches 100  \
                   --num-workers 8
done               

echo "finished"