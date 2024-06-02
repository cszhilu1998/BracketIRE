#!/bin/bash

echo "Start to train the model...."
dataroot="/Dataset/MultiExpo/Realdata/"

device='0,1'
name="real_try"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python ./real_train.py \
    --dataset_name real    --model real            --name $name           --lr_policy cosine      --dataroot $dataroot   \
    --patch_size 128       --niter 10              --save_imgs False      --lr  7.5e-5   \
    --batch_size 8         --print_freq 500        --calc_metrics True    --weight_decay 0.01 \
    --gpu_ids $device      --save_epoch_freq 1     --self_weight 1        -j 8   | tee $LOG     
