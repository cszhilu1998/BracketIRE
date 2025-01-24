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
from process import process
from unprocess import unprocess
from tqdm import tqdm
import random
import imageio



def bulid_model(modelDir='train_log', device=None):
    args_modelDir = modelDir

    from train_log.RIFE_HDv3 import Model
    model = Model(device)
    model.load_model(args_modelDir, -1)
    print("Loaded v3.x HD model.")

    model.eval()
    model.device()

    return model


def frame_inter(img0, img1, model, exp=5):
    args_exp = exp

    img0 = torch.clamp(img0.permute(2, 0, 1) / 65535., 0, 1).unsqueeze(0)
    img1 = torch.clamp(img1.permute(2, 0, 1) / 65535., 0, 1).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_list = [img0, img1]
    for i in range(args_exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

    img_list_out = []
    for img in img_list[1:len(img_list)-1]:
        img_list_out.append(img.squeeze(0).permute(1, 2, 0)[:h, :w] * 65535)
        
    del img_list, img0, img1, tmp
    return img_list_out


def mu_tonemap(hdr_image, mu=5000):
    mu = torch.tensor(mu).to(hdr_image.device)
    return torch.log(1 + mu * hdr_image) / torch.log(1 + mu)


def gamma(pre_img):
    """Parameters are from HdM-HDR-2014 dataset (https://www.hdm-stuttgart.de/vmlab/hdm-hdr-2014/#FTPdownload)."""
    Mask = lambda x: (x>0.0031308).float()
    sRGBDeLinearize = lambda x,m: m * (1.055 * (m * x) ** (1/2.4) - 0.055) + (1-m) * (12.92 * x)
    return  sRGBDeLinearize(pre_img, Mask(pre_img))


def gamma_reverse(pre_img): 
    """Parameters are from HdM-HDR-2014 dataset (https://www.hdm-stuttgart.de/vmlab/hdm-hdr-2014/#FTPdownload)."""
    Mask = lambda x: (x>0.04045).float()
    sRGBLinearize = lambda x,m: m * ((m * x + 0.055) / 1.055) ** 2.4 + (1-m) * (x / 12.92)
    return  sRGBLinearize(pre_img, Mask(pre_img))


def split_name(path):
   name = path.split('/')[-1].split('_')[-1][:-4]
   return name


def read_11_paths(img_path, frame_num):
    paths = []
    img_names = sorted(glob.glob(img_path + '*.exr'))
    for i in range(0, len(img_names)//frame_num*frame_num, frame_num):
        paths.append(list(img_names[i:i+frame_num]))
    return paths


def read_exr(img_exr_path, device=None, expo=-4):
    AlexaWideGamut2sRGB = np.array([[1.617523436306807,  -0.070572740897816,  -0.021101728042793], 
                                    [-0.537286622188294,   1.334613062330328,  -0.226953875218266],
                                    [-0.080236814118512,  -0.264040321432512,   1.248055603261060]])
    pre_img = np.array(imageio.imread(img_exr_path, 'exr'))
    pre_img = np.clip(pre_img, 0, None)
    pre_img =  (pre_img * 2.**expo) @ AlexaWideGamut2sRGB
    pre_img = torch.from_numpy(pre_img.astype(np.float32)).to(device)
    return gamma(pre_img), pre_img.max()


def get_rgb2raw(img, features=None, device=None):
    deg_img, features = unprocess(img, features, device)
    return deg_img, features


def get_raw2rgb(img, features, demosaic='default', lineRGB=False):
    img = img.unsqueeze(0)
    device = img.device 
    deg_img = process(img, features['red_gain'].unsqueeze(0), features['blue_gain'].unsqueeze(0), 
                      features['cam2rgb'].unsqueeze(0), demosaic, lineRGB)
    deg_img = deg_img.squeeze(0)
    return deg_img
