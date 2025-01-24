# coding=utf-8
import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.nn as nn
import torch.distributions as tdist
import glob
from function import *
from unprocess import add_noise, random_iso1600near_levels
from tqdm import tqdm
import random
import imageio


''' Generating paired data for BracketIRE task '''

def write_img(raw_img, meta_data, folder_dir, name='', raw_max=1):
    
    if name == 'gt':
        np.save(final_dir + 'alignratio.npy', raw_max.cpu().numpy())

        raw_img = raw_img * raw_max
        np.save(folder_dir + 'raw_' + name + '.npy', raw_img.clone().cpu().numpy())
        
        raw_img = torch.clamp(raw_img/16., 0, 1)
        rgb_img = get_raw2rgb(raw_img, meta_data, demosaic='net', lineRGB=True) # menon2007, net
        rgb_img = torch.clamp(mu_tonemap(rgb_img)*65535, 0.0, 65535.0).cpu().numpy().astype(np.uint16)
        cv2.imwrite(folder_dir + 'rgb_vis_' + name + '.png', rgb_img[..., ::-1])
    
    else:
        raw_img = torch.clamp(raw_img * (2**10-1), 0, 2**10-1)
        np.save(folder_dir + 'raw_' + name + '.npy', raw_img.cpu().numpy().round().astype(np.uint16))

        rgb_img = get_raw2rgb(raw_img / (2**10-1), meta_data, demosaic='net', lineRGB=False) # menon2007, net

        rgb_img = torch.clamp(rgb_img * 255.0, 0.0, 255.0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(folder_dir + 'rgb_vis_' + name + '.png', rgb_img[..., ::-1])


if __name__ == '__main__':
    read_path = '/dataset/HDM-HDR-2014/HdM-HDR-2014_Original-HDR-Camera-Footage/'
    # Download from https://www.hdm-stuttgart.de/vmlab/hdm-hdr-2014/#FTPdownload
    write_root = '/dataset/BracketIRE/'
    device = torch.device("cuda:0")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model = bulid_model(modelDir='train_log', device=device)
    frame_num = 12

    # print(sorted(os.listdir(read_path)))
    img_paths =  ['beerfest_lightshow_01', 'beerfest_lightshow_02', 'beerfest_lightshow_02_reconstruction_update_2015', 
                 'beerfest_lightshow_03', 'beerfest_lightshow_04', 'beerfest_lightshow_04_reconstruction_update_2015', 
                 'beerfest_lightshow_05', 'beerfest_lightshow_06', 'beerfest_lightshow_07', 
                 'bistro_01', 'bistro_02', 'bistro_03',
                 'carousel_fireworks_01', 'carousel_fireworks_02', 'carousel_fireworks_03', 
                 'carousel_fireworks_04', 'carousel_fireworks_05', 'carousel_fireworks_06', 
                 'carousel_fireworks_07', 'carousel_fireworks_08', 'carousel_fireworks_09',
                 'cars_closeshot', 'cars_fullshot', 'cars_longshot',
                 'fireplace_01', 'fireplace_02', 'fishing_closeshot', 
                 'showgirl_01', 'showgirl_02', 'smith_hammering',
                 'smith_welding','fishing_longshot', 'hdr_testimage', 
                 'poker_fullshot', 'poker_travelling_slowmotion']

    for path in img_paths:
        img_path = read_path + path + '/'

        folder_dir = write_root + path + '/'
        os.makedirs(folder_dir, exist_ok=True)
        print(folder_dir)

        frame_paths = read_11_paths(img_path, frame_num)        

        for frame_path in tqdm(frame_paths):
            list_imgs = []

            final_dir = folder_dir + split_name(frame_path[0]) + '/'
            os.makedirs(final_dir, exist_ok=True)

            pre_img, raw_max = read_exr(frame_path[0], device=device)
            pre_img_gt = torch.clamp(gamma(gamma_reverse(pre_img)/raw_max), 0, 1)
            # print(pre_img_gt.max(), pre_img_gt.min())
            clean_raw, features = get_rgb2raw(pre_img_gt, features=None, device=device)

            H, W, C = clean_raw.size()
            write_img(clean_raw, features, final_dir, 'gt', raw_max)

            meta = {}
            for key in features:
                meta[key] = features[key].cpu().numpy().astype(np.float32)
            np.save(final_dir + 'metadata.npy', meta)
            
            del clean_raw, pre_img_gt, meta

            curr_max = 65535 / pre_img.max()
            pre_img = pre_img * curr_max

            list_imgs.append(pre_img)

            for i in range(frame_num-1):
                next_img = read_exr(frame_path[i+1], device=device)[0] * curr_max
                list_imgs.extend(frame_inter(pre_img, next_img, model, exp=5))
                list_imgs.append(next_img)
                pre_img = next_img

            del pre_img, next_img

            start = 0
            for n in range(5):
                raw = torch.zeros([H, W, C], dtype=torch.float, device=device)
                m = 4 ** n

                for i in range(m):
                    img_ldr = list_imgs[start + i] / curr_max 
                    img_ldr = torch.clamp(gamma(gamma_reverse(img_ldr)*4**(n-2)), 0, 1)
                    gt_raw, _ = get_rgb2raw(img_ldr, features, device) 
                    raw = raw + gt_raw   
                
                raw = torch.clamp(raw / m, 0, 1)
                shot_noise, read_noise = random_iso1600near_levels(torch.rand(1, device=device)[0])
                raw = add_noise(raw, shot_noise, read_noise)
                raw = torch.clamp(raw, 0, 1)

                write_img(raw, features, final_dir, str(m))
                del gt_raw, img_ldr, raw
                
                start = start + m
            
            del list_imgs 
            
