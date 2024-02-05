#!/bin/bash
echo "Start to test the model...."
device="0"


dataroot="/Data/bracketire_plus/"  # including 'Train' and 'NTIRE_Val' floders
name="bracketire_plus"

python test.py \
    --dataset_name bracketireplus   --model  tmrnetplus     --name $name              --dataroot $dataroot  \
    --load_iter 400                 --save_imgs True        --calc_metrics False      --gpu_id $device  -j 8   
