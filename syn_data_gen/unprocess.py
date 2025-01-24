# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unprocesses sRGB images into realistic raw data.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import torch.distributions as tdist


def random_ccm(device):
    """Obtained from Xiaomi 10S."""
    xyz2cam = torch.FloatTensor([[0.734375, -0.125, -0.0859375],
                                [-0.2734375, 1.109375, 0.140625], 
                                [-0.015625, 0.21875, 0.40625]])

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                [0.2126729, 0.7151522, 0.0721750],
                                [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam.to(device), rgb2xyz.to(device))

    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    return rgb2cam


def random_gains(device):
    """Generates random gains for brightening and white balance."""
    """Parameters are from Xiaomi 10S."""
    # Red and blue gains represent white balance.
    red_gain  =  torch.FloatTensor(1).uniform_(1/0.91, 1/0.40)
    blue_gain =  torch.FloatTensor(1).uniform_(1/0.75, 1/0.39)
    rgb_gain = torch.FloatTensor([1])
    return rgb_gain.to(device), red_gain.to(device), blue_gain.to(device)


def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = torch.clamp(image, min=0.0, max=1.0)
    out   = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0) 
    return out


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    """Parameters are from HdM-HDR-2014 dataset (https://www.hdm-stuttgart.de/vmlab/hdm-hdr-2014/#FTPdownload)."""
    # Clamps to prevent numerical instability of gradients near zero.
    Mask = lambda x: (x>0.04045).float()
    sRGBLinearize = lambda x,m: m * ((m * x + 0.055) / 1.055) ** 2.4 + (1-m) * (x / 12.92)
    return  sRGBLinearize(image, Mask(image))


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    shape = image.size()
    image = torch.reshape(image, [-1, 3])
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    out   = torch.reshape(image, shape)
    return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain, device):
    """Inverts gains while safely handling saturated pixels."""
    gains = torch.stack((1.0 / red_gain, torch.tensor([1.0]).to(device), 1.0 / blue_gain))
    gains = gains.to(device).squeeze()
    gains = gains[None, None, :]
    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray  = torch.mean(image, dim=-1, keepdim=True)
    inflection = 0.9
    mask  = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
    safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    out   = image * safe_gains
    return out


def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.size()
    red   = image[0::2, 0::2, 0]
    green_red  = image[0::2, 1::2, 1]
    green_blue = image[1::2, 0::2, 1]
    blue = image[1::2, 1::2, 2]
    out  = torch.stack((red, green_red, green_blue, blue), dim=-1)
    out  = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
    return out


def unprocess(image, features=None, device=None):
    """Unprocesses an image from sRGB to realistic raw data."""

    if features == None:
    # Randomly creates image metadata.
        rgb2cam = random_ccm(device)
        cam2rgb = torch.inverse(rgb2cam)
        rgb_gain, red_gain, blue_gain = random_gains(device)
    else:
        rgb2cam = features['rgb2cam']
        cam2rgb = features['cam2rgb']
        rgb_gain = features['rgb_gain']
        red_gain = features['red_gain']
        blue_gain = features['blue_gain']

    # Approximately inverts global tone mapping.
    # image = inverse_smoothstep(image)

    # Inverts gamma compression.
    image = gamma_expansion(image)

    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)

    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain, device)

    # Clips saturated pixels.
    image = torch.clamp(image, min=0.0, max=1.0)

    # Applies a Bayer mosaic.
    image = mosaic(image)

    metadata = {
        'rgb2cam': rgb2cam,
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return image, metadata


def add_noise(image, shot_noise, read_noise):
    var = image * shot_noise + read_noise
    noise = tdist.Normal(loc=torch.zeros_like(var), scale=torch.sqrt(var)).sample()
    out = image + noise                    
    return out


def random_iso1600_levels(device):
    """Noise Parameters are from Xiaomi 10S (ISO=1600)."""
    shot_noise = torch.tensor(0.00242237221).to(device)
    log_read_noise = torch.log(torch.tensor(1.790097494e-05)).to(device)

    n = tdist.Normal(loc=torch.tensor([0.0], device=device), 
                    scale=torch.tensor([0.30], device=device))

    log_read_noise = log_read_noise + n.sample()
    read_noise     = torch.exp(log_read_noise)

    return shot_noise, read_noise


def random_iso1600near_levels(noise_level):
    """Noise Parameters are from Xiaomi 10S (Near ISO=1600)."""
    log_min_shot_noise = torch.log(torch.tensor(0.0012)).to(noise_level.device)
    log_max_shot_noise = torch.log(torch.tensor(0.0048)).to(noise_level.device)
    log_shot_noise = log_min_shot_noise + noise_level * (log_max_shot_noise - log_min_shot_noise)
    shot_noise = torch.exp(log_shot_noise)

    line = lambda x: 1.869 * x + 0.3276
    n = tdist.Normal(loc=torch.tensor([0.0], device=noise_level.device), 
                    scale=torch.tensor([0.30], device=noise_level.device))

    log_read_noise = line(log_shot_noise) + n.sample()
    read_noise     = torch.exp(log_read_noise)

    return shot_noise, read_noise

