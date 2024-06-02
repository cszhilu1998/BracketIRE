#!/bin/bash
echo "Start to test the model...."
device="0"

dataroot="/Dataset/MultiExpo/Realdata/"
name="real_plus"
  
python ./real_test.py \
    --dataset_name real   --model  realplus      --name $name   --dataroot $dataroot   -j 8  \
    --load_iter 10        --save_imgs True       --chop True    --gpu_id $device  

python ./real_metrics.py    --name $name     --save_img output_real_vispng_10    --device $device


