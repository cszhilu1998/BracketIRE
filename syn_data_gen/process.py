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

"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import os
import demosaic_bayer


def apply_gains(bayer_images, red_gains, blue_gains):
  """Applies white balance gains to a batch of Bayer images."""
  red_gains = red_gains.squeeze(1)
  blue_gains= blue_gains.squeeze(1)
  green_gains  = torch.ones_like(red_gains)
  gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], dim=-1)
  gains = gains[:, None, None, :]
  outs  = bayer_images * gains
  return outs


def demosaic(bayer_images):
    def SpaceToDepth_fact2(x):
        # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        bs = 2
        N, C, H, W = x.size()
        x = x.view(N, C, H // bs, bs, W // bs, bs)      # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()    # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (bs ** 2), H // bs, W // bs)  # (N, C*bs^2, H//bs, W//bs)
        return x
    def DepthToSpace_fact2(x):
        # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        bs = 2
        N, C, H, W = x.size()
        x = x.view(N, bs, bs, C // (bs ** 2), H, W)     # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()    # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (bs ** 2), H * bs, W * bs)   # (N, C//bs^2, H * bs, W * bs)
        return x

    """Bilinearly demosaics a batch of RGGB Bayer images."""

    shape = bayer_images.size()
    shape = [shape[1] * 2, shape[2] * 2]

    red = bayer_images[Ellipsis, 0:1]
    upsamplebyX = nn.Upsample(size=shape, mode='bilinear', align_corners=False)
    red = upsamplebyX(red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_red = bayer_images[Ellipsis, 1:2]
    green_red = torch.flip(green_red, dims=[1]) # Flip left-right
    green_red = upsamplebyX(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    green_red = torch.flip(green_red, dims=[1]) # Flip left-right
    green_red = SpaceToDepth_fact2(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_blue = bayer_images[Ellipsis, 2:3]
    green_blue = torch.flip(green_blue, dims=[0]) # Flip up-down
    green_blue = upsamplebyX(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    green_blue = torch.flip(green_blue, dims=[0]) # Flip up-down
    green_blue = SpaceToDepth_fact2(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_at_red = (green_red[Ellipsis, 0] + green_blue[Ellipsis, 0]) / 2
    green_at_green_red = green_red[Ellipsis, 1]
    green_at_green_blue = green_blue[Ellipsis, 2]
    green_at_blue = (green_red[Ellipsis, 3] + green_blue[Ellipsis, 3]) / 2

    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = DepthToSpace_fact2(torch.stack(green_planes, dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    blue = bayer_images[Ellipsis, 3:4]
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])
    blue = upsamplebyX(blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])

    rgb_images = torch.cat([red, green, blue], dim=-1)
    return rgb_images


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images[:, :, :, None, :]
    ccms   = ccms[:, None, None, :, :]
    outs   = torch.sum(images * ccms, dim=-1)
    return outs


def gamma_compression(images):
    """Converts from linear to gamma space."""
    """Parameters are from HdM-HDR-2014 dataset (https://www.hdm-stuttgart.de/vmlab/hdm-hdr-2014/#FTPdownload)."""
    Mask = lambda x: (x>0.0031308).float()
    sRGBDeLinearize = lambda x,m: m * (1.055 * (m * x) ** (1/2.4) - 0.055) + (1-m) * (12.92 * x)
    return  sRGBDeLinearize(images, Mask(images))



def process(bayer_images, red_gains, blue_gains, cam2rgbs, demosaic_type, lineRGB):
    # print(bayer_images.shape, red_gains.shape, cam2rgbs.shape)
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # White balance.
    bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    
    if demosaic_type == 'default':
        images = demosaic(bayer_images)
    elif demosaic_type == 'menon2007':
        bayer_images = flatten_raw_image(bayer_images.squeeze(0))
        images = demosaicing_CFA_Bayer_Menon2007(bayer_images.cpu().numpy(), 'RGGB')
        images = torch.from_numpy(images).unsqueeze(0).to(red_gains.device)
    elif demosaic_type == 'net':
        bayer_images = flatten_raw_image(bayer_images.squeeze(0)).cpu().numpy()
        bayer = np.power(np.clip(bayer_images.astype(dtype=np.float32), 0, 1), 1 / 2.2)
        pretrained_model_path = os.path.dirname(__file__) + "/model.bin"
        demosaic_net = demosaic_bayer.get_demosaic_net_model(pretrained=pretrained_model_path, device=red_gains.device, 
                                                             cfa='bayer', state_dict=True)
        rgb = demosaic_bayer.demosaic_by_demosaic_net(bayer=bayer, cfa='RGGB', 
                                                      demosaic_net=demosaic_net, device=red_gains.device)
        images = np.power(np.clip(rgb, 0, 1), 2.2) 
        images = torch.from_numpy(images).unsqueeze(0).to(red_gains.device)

    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    if not lineRGB:
        images = gamma_compression(images)
    return images


def flatten_raw_image(im_raw_4ch):  # HxWx4
    """ unpack a 4-channel tensor into a single channel bayer image"""
    if isinstance(im_raw_4ch, np.ndarray):
        im_out = np.zeros_like(im_raw_4ch, shape=(im_raw_4ch.shape[0] * 2, im_raw_4ch.shape[1] * 2))
    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros((im_raw_4ch.shape[0] * 2, im_raw_4ch.shape[1] * 2), dtype=im_raw_4ch.dtype)
    else:
        raise Exception

    im_out[0::2, 0::2] = im_raw_4ch[:, :, 0]
    im_out[0::2, 1::2] = im_raw_4ch[:, :, 1]
    im_out[1::2, 0::2] = im_raw_4ch[:, :, 2]
    im_out[1::2, 1::2] = im_raw_4ch[:, :, 3]

    return im_out  # HxW