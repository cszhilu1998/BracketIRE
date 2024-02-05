#!/bin/bash
echo "Start to test the model...."
device="0"


dataroot="/Data/bracketire/"  # including 'Train' and 'NTIRE_Val' floders
name="bracketire"

python test.py \
    --dataset_name bracketire   --model  tmrnet       --name $name          --dataroot $dataroot  \
    --load_iter 400             --save_imgs True      --calc_metrics False  --gpu_id $device  -j 8   



