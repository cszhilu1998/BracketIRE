#!/bin/bash
echo "Start to train the model...."
dataroot="/Data/bracketire_plus/"  # including 'Train' and 'NTIRE_Val' floders

device='0'
name="track2"
load_path="xx" # path of pre-trained model for BracketIRE task

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --dataset_name bracketireplus    --model tmrnetplus    --name $name            --lr_policy cosine_warmup      \
    --patch_size 64                  --niter 400           --save_imgs False       --lr 1e-4          --dataroot $dataroot   \
    --batch_size 8                   --print_freq 500      --calc_metrics True     --weight_decay 0.01 \
    --gpu_ids $device     -j 8       --load_path  $load_path    | tee $LOG 

