import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from collections import OrderedDict
import torch.nn.functional as F
from util.util import SSIM
import random
import torch.distributions as tdist
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange
import torchvision.ops as ops
import numbers
from torch.nn.modules.utils import _triple
import math
import cv2



# Augument
def augment_func(img, hflip, vflip, rot90):  # CxHxW
    if hflip:   img = torch.flip(img, dims=[-1])
    if vflip:   img = torch.flip(img, dims=[-2])
    if rot90:   img = img.transpose(img.ndim-1, img.ndim-2)
    return img


def augment(*imgs):  # CxHxW
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    return (augment_func(img, hflip, vflip, rot90) for img in imgs)


# Pack Images
def pack_raw_image(im_raw):  # GT: N x 1 x H x W ; RAWs: N x T x H x W
    """ Packs a single channel bayer image into 4 channel tensor, where channels contain R, G, G, and B values"""
    im_out = torch.zeros([im_raw.shape[0], im_raw.shape[1], 4, im_raw.shape[2] // 2, im_raw.shape[3] // 2], 
                         dtype=im_raw.dtype, device=im_raw.device)

    im_out[..., 0, :, :] = im_raw[:, :, 0::2, 0::2]
    im_out[..., 1, :, :] = im_raw[:, :, 0::2, 1::2]
    im_out[..., 2, :, :] = im_raw[:, :, 1::2, 0::2]
    im_out[..., 3, :, :] = im_raw[:, :, 1::2, 1::2]
     
    # GT: N x 1 x 4 x (H//2) x (W//2)
    # RAWs: N x T x 4 x (H//2) x (W//2)
    return im_out


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.niter,
                                                   eta_min=1e-6)
    elif opt.lr_policy == 'cosine_warmup':
        scheduler = CosineLRScheduler(optimizer, 
                                      t_initial=opt.niter, 
                                      lr_min=1e-6,
                                      warmup_t=5,
                                      warmup_lr_init=1e-5)
    else:
        return NotImplementedError('lr [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
                or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, b=init_gain)
            elif init_type == 'constant':
                init.constant_(m.weight.data, 0.0)
            else:
                raise NotImplementedError('[%s] is not implemented' % init_type)
        elif hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_type != 'default' and init_type is not None:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=x.dtype),
        torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def load_spynet(net, path):
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(path)

        print('loading the model from %s' % (path))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        net_state = net.state_dict()
        is_loaded = {n:False for n in net_state.keys()}
        for name, param in state_dict.items():
            name = name.replace('basic_module.0.conv', 'basic_module.0')
            name = name.replace('basic_module.1.conv', 'basic_module.2')
            name = name.replace('basic_module.2.conv', 'basic_module.4')
            name = name.replace('basic_module.3.conv', 'basic_module.6')
            name = name.replace('basic_module.4.conv', 'basic_module.8')
            if name in net_state:
                try:
                    net_state[name].copy_(param)
                    is_loaded[name] = True
                except Exception:
                    print('While copying the parameter named [%s], '
                            'whose dimensions in the model are %s and '
                            'whose dimensions in the checkpoint are %s.'
                            % (name, list(net_state[name].shape),
                                list(param.shape)))
                    raise RuntimeError
            else:
                print('Saved parameter named [%s] is skipped' % name)
        mark = True
        for name in is_loaded:
            if not is_loaded[name]:
                print('Parameter named [%s] is not initialized' % name)
                mark = False
        if mark:
            print('All parameters are initialized using [%s]' % path)


'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, nn.Module):
        return args
    modules = OrderedDict()
    if isinstance(args, OrderedDict):
        for k, v in args.items():
            modules[k] = seq(v)
        return nn.Sequential(modules)
    assert isinstance(args, (list, tuple))
    return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
         output_padding=0, dilation=1, groups=1, bias=True,
         padding_mode='zeros', mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'X':
            assert in_channels == out_channels
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias,
                                        dilation=dilation,
                                        padding_mode=padding_mode))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'i':
            L.append(nn.InstanceNorm2d(out_channels))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'S':
            L.append(nn.Sigmoid())
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return seq(*L)
