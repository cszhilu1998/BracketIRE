import cv2
import numpy as np
import random
import torch
from scipy import ndimage
from scipy.interpolate import interp2d
from .unprocess import unprocess, random_noise_levels, add_noise
from .process import process
from PIL import Image



# def get_rgb2raw2rgb(img):
#     img = torch.from_numpy(np.array(img)) / 255.0
#     deg_img, features = unprocess(img)
#     shot_noise, read_noise = random_noise_levels()
#     deg_img = add_noise(deg_img, shot_noise, read_noise)
#     deg_img = deg_img.unsqueeze(0)
#     features['red_gain'] = features['red_gain'].unsqueeze(0)
#     features['blue_gain'] = features['blue_gain'].unsqueeze(0)
#     features['cam2rgb'] = features['cam2rgb'].unsqueeze(0)
#     deg_img = process(deg_img, features['red_gain'], features['blue_gain'], features['cam2rgb'])
#     deg_img = deg_img.squeeze(0)
#     deg_img = torch.clamp(deg_img * 255.0, 0.0, 255.0).numpy()
#     deg_img = deg_img.astype(np.uint8)
#     return Image.fromarray(deg_img)


# def get_rgb2raw_noise(img, noise_level, features=None):
#     # img = np.transpose(img, (1, 2, 0))
#     img = torch.from_numpy(np.array(img)) / 255.0

#     deg_img, features = unprocess(img, features)
#     shot_noise, read_noise = random_noise_levels(noise_level)
#     deg_img_noise = add_noise(deg_img, shot_noise, read_noise)
#     # deg_img_noise = torch.clamp(deg_img_noise, min=0.0, max=1.0)

#     # deg_img = np.transpose(deg_img, (2, 0, 1))
#     # deg_img_noise = np.transpose(deg_img_noise, (2, 0, 1))
#     return deg_img_noise, features


def get_rgb2raw(img, features=None):
    # img = np.transpose(img, (1, 2, 0))
    device = img.device
    deg_img, features = unprocess(img, features, device)
    return deg_img, features


def get_raw2rgb(img, features, demosaic='net', lineRGB=False):
    # img = np.transpose(img, (1, 2, 0))
    # img = torch.from_numpy(np.array(img))
    img = img.unsqueeze(0)
    device = img.device 
    deg_img = process(img, features['red_gain'].to(device), features['blue_gain'].to(device), 
                      features['cam2rgb'].to(device), demosaic, lineRGB)
    deg_img = deg_img.squeeze(0)
    # deg_img = torch.clamp(deg_img * 255.0, 0.0, 255.0).numpy()
    # deg_img = deg_img.astype(np.uint8)
    return deg_img


# def pack_raw_image(im_raw):  # HxW
#     """ Packs a single channel bayer image into 4 channel tensor, where channels contain R, G, G, and B values"""
#     if isinstance(im_raw, np.ndarray):
#         im_out = np.zeros_like(im_raw, shape=(4, im_raw.shape[0] // 2, im_raw.shape[1] // 2))
#     elif isinstance(im_raw, torch.Tensor):
#         im_out = torch.zeros((4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype)
#     else:
#         raise Exception

#     im_out[0, :, :] = im_raw[0::2, 0::2]
#     im_out[1, :, :] = im_raw[0::2, 1::2]
#     im_out[2, :, :] = im_raw[1::2, 0::2]
#     im_out[3, :, :] = im_raw[1::2, 1::2]
#     return im_out  # 4xHxW


# def flatten_raw_image(im_raw_4ch):  # 4xHxW
#     """ unpack a 4-channel tensor into a single channel bayer image"""
#     if isinstance(im_raw_4ch, np.ndarray):
#         im_out = np.zeros_like(im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2))
#     elif isinstance(im_raw_4ch, torch.Tensor):
#         im_out = torch.zeros((im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype)
#     else:
#         raise Exception

#     im_out[0::2, 0::2] = im_raw_4ch[0, :, :]
#     im_out[0::2, 1::2] = im_raw_4ch[1, :, :]
#     im_out[1::2, 0::2] = im_raw_4ch[2, :, :]
#     im_out[1::2, 1::2] = im_raw_4ch[3, :, :]

#     return im_out  # HxW