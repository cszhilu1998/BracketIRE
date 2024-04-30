#!/bin/bash

echo "Start to train the model...."
dataroot="/Data/dataset/MultiExpo/Syn_Plus/" 

device='0,1'
name="syn_plus_try"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python ./syn_train.py \
    --dataset_name synplus           --model synplus       --name $name            --lr_policy cosine_warmup      \
    --patch_size 64                  --niter 400           --save_imgs False       --lr 1e-4          --dataroot $dataroot   \
    --batch_size 8                   --print_freq 500      --calc_metrics True     --weight_decay 0.01 \
    --load_path ./ckpt/syn/TMRNet_model_400.pth   \
    --gpu_ids $device     -j 8     | tee $LOG 

    # --load_path 'path of pre-trained model for synthetic BracketIRE task'