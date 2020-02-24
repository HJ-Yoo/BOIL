#!/bin/bash

python miniimagenet_train.py --data_path=/home/osilab7/hdd/mini-imagenet-jpg/ \
                             --device=cuda:1 \
                             --epoch=60000 \
                             --n_way=5 \
                             --k_spt=5 \
                             --k_qry=15 \
                             --imgsz=84 \
                             --imgc=3 \
                             --task_num=4 \
                             --meta_lr=1e-3 \
                             --update_lr=1e-2 \
                             --update_step=5 \
                             --update_step_test=10
echo "finished"
