#!/bin/bash
echo "Start to test the model...."
device="0"

dataroot="/Data/dataset/MultiExpo/Syn_Plus/"
name="syn_plus"

# Note that this evaluation method is different from that of 'Bracketing Image Restoration and Enhancement Challenges on NTIRE 2024'.
# Here, we exclude invalid pixels around the image for evaluation. 
# It is more reasonable. 
# We recommend that you use this evaluation method in your future work.

python ./syn_test.py  \
    --dataset_name synplus        --model  synplus      --name $name           --dataroot $dataroot  \
    --load_iter 400               --save_imgs True      --gpu_id $device       -j 8   

python ./syn_metrics.py  \
    --dataroot $dataroot          --name $name          --save_img  output_vispng_400    \
    --plus True                   --device $device