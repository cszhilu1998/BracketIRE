import os
import sys
# sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
import pdb
import copy
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BayerNetwork(nn.Module):
    """Released version of the network, best quality.

    This model differs from the published description. It has a mask/filter split
    towards the end of the processing. Masks and filters are multiplied with each
    other. This is not key to performance and can be ignored when training new
    models from scratch.
    """
    def __init__(self, depth=15, width=64):
        super(BayerNetwork, self).__init__()

        self.depth = depth
        self.width = width

        # self.debug_layer = nn.Conv2d(3, 4, 2, stride=2)
        # self.debug_layer1 =nn.Conv2d(in_channels=4,out_channels=64,kernel_size=3,stride=1,padding=1)
        # self.debug_layer2 =nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        # self.debug_layer3 =nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)

        layers = OrderedDict([
            ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),  # Downsample 2x2 to re-establish translation invariance.
        ])                                                  #
                                                            # the output of 'pack_mosaic' will be half width and height of the input
                                                            # [batch_size, 4, h/2, w/2] = pack_mosaic ( [batch_size, 3, h, w] )

        for i in range(depth):
            #num of in and out neurons in each layers
            n_out = width
            n_in = width

            if i == 0:                          # the 1st layer in main_processor
                n_in = 4
            if i == depth-1:                    # the last layer in main_processor
                n_out = 2*width

            # layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
            layers["conv{}".format(i + 1)] = nn.Conv2d(n_in, n_out, 3,stride=1,padding=1)          
            # padding is set to be 1 so that the h and w won't change after conv2d (using kernal size 3)
            layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)


        # main conv layer
        self.main_processor = nn.Sequential(layers)
        # residual layer
        self.residual_predictor = nn.Conv2d(width, 12, 1)
        # upsample layer
        self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)

        # full-res layer
        self.fullres_processor = nn.Sequential(OrderedDict([
            # ("post_conv", nn.Conv2d(6, width, 3)),
            ("post_conv", nn.Conv2d(6, width, 3,stride=1,padding=1)),                                
            # padding is set to be 1 so that the h and w won't change after conv2d (using kernal size 3)
            ("post_relu", nn.ReLU(inplace=True)),
            ("output", nn.Conv2d(width, 3, 1)),
        ]))


    # samples structure
    #   sample = {
    #         "mosaic": mosaic,                               
    #         # model input   [batch_size, 3, h,w]. unknown pixels are set to 0.
    #         "mask": mask,
    #         # "noise_variance": np.array([std]),
    #         "target": im,                                   
    #         # model output  [m,n,3]
    #     }
    def forward(self, samples):

        mosaic = samples["mosaic"]                                                  
        # [batch_size, 3, h, w]

        features = self.main_processor(mosaic)                                      
        # [batch_size, self.width*2, hf,wf]

        filters, masks = features[:, :self.width], features[:, self.width:]

        filtered = filters * masks                                                  
        # [batch_size, self.width, hf,wf]

        residual = self.residual_predictor(filtered)                                
        # [batch_size, 12, hf, wf]

        upsampled = self.upsampler(residual)                                        
        # [batch_size, 3, hf*2, wf*2]. upsampled will be 2x2 upsample of residual using ConvTranspose2d()

        # crop original mosaic to match output size
        cropped = crop_like(mosaic, upsampled)

        # Concated input samples and residual for further filtering
        packed = torch.cat([cropped, upsampled], 1)

        output = self.fullres_processor(packed)

        return output


class Converter(object):
  def __init__(self, pretrained_dir, model_type):
    self.basedir = pretrained_dir

  def convert(self, model):
    for n, p in model.named_parameters():
      name, tp = n.split(".")[-2:]

      old_name = self._remap(name)
      # print(old_name, "->", name)

      if tp == "bias":
        idx = 1
      else:
        idx = 0
      path = os.path.join(self.basedir, "{}_{}.npy".format(old_name, idx))
      data = np.load(path)
      # print(name, tp, data.shape, p.shape)

      # Overwiter
      # print(p.mean().item(), p.std().item())
      # import ipdb; ipdb.set_trace()
      # print(name, old_name, p.shape, data.shape)
      p.data.copy_(torch.from_numpy(data))
      # print(p.mean().item(), p.std().item())

  def _remap(self, s):
    if s == "pack_mosaic":
      return "pack_mosaick"
    if s == "residual_predictor":
      return "residual"
    if s == "upsampler":
      return "unpack_mosaick"
    if s == "post_conv":
      return "post_conv1"
    return s


def crop_like(src, tgt):
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)
    crop = (src_sz[2:4]-tgt_sz[2:4]) // 2
    if (crop > 0).any():
        return src[:, :, crop[0]:src_sz[2]-crop[0], crop[1]:src_sz[3]-crop[1], ...]
    else:
        return src


def get_modules(params):
    params = copy.deepcopy(params)  # do not touch the original

    # get the model name from the input params
    model_name = params.pop("model", None)

    if model_name is None:
        raise ValueError("model has not been specified!")

    # get the model structure by model_name
    return getattr(sys.modules[__name__], model_name)(**params)


def get_demosaic_net_model(pretrained, device, cfa='bayer', state_dict=False):
    '''
        get demosaic network
    :param pretrained:
        path to the demosaic-network model file [string]
    :param device:
        'cuda:0', e.g.
    :param state_dict:
        whether to use a packed state dictionary for model weights
    :return:
        model_ref: demosaic-net model

    '''

    model_ref = get_modules({"model": "BayerNetwork"})  # load model coefficients if 'pretrained'=True
    if not state_dict:
        cvt = Converter(pretrained, "BayerNetwork")
        cvt.convert(model_ref)
        for p in model_ref.parameters():
            p.requires_grad = False
        model_ref = model_ref.to(device)
    else:
        model_ref.load_state_dict(torch.load(pretrained))
        model_ref = model_ref.to(device)

        model_ref.eval()

    return model_ref


def demosaic_by_demosaic_net(bayer, cfa, demosaic_net, device):
    '''
        demosaic the bayer to get RGB by demosaic-net. The func will covnert the numpy array to tensor for demosaic-net,
        after which the tensor will be converted back to numpy array to return.

    :param bayer:
        [m,n]. numpy float32 in the rnage of [0,1] linear bayer
    :param cfa:
        [string], 'RGGB', e.g. only GBRG, RGGB, BGGR or GRBG is supported so far!
    :param demosaic_net:
        demosaic_net object
    :param device:
        'cuda:0', e.g.

    :return:
        [m,n,3]. np array float32 in the rnage of [0,1]

    '''


    assert (cfa == 'GBRG') or (cfa == 'RGGB') or (cfa == 'GRBG') or (cfa == 'BGGR'), 'only GBRG, RGGB, BGGR, GRBG are supported so far!'

    # if the bayer resolution is too high (more than 1000x1000,e.g.), may cause memory error.

    bayer = np.clip(bayer ,0 ,1)
    bayer = torch.from_numpy(bayer).float()
    bayer = bayer.to(device)
    bayer = torch.unsqueeze(bayer, 0)
    bayer = torch.unsqueeze(bayer, 0)

    with torch.no_grad():
        rgb = predict_rgb_from_bayer_tensor(bayer, cfa=cfa, demosaic_net=demosaic_net, device=device)

    rgb = rgb.detach().cpu()[0].permute(1, 2, 0).numpy()  # torch tensor -> numpy array
    # rgb = np.clip(rgb, 0, 1)

    return rgb


def predict_rgb_from_bayer_tensor(im,cfa,demosaic_net,device):
    '''
        predict the RGB imgae from bayer pattern mosaic using demosaic net

    :param im:
        [batch_sz, 1, m,n] tensor. the bayer pattern mosiac.

    :param cfa:
        the cfa layout. the demosaic net is trained w/ GRBG. If the input is other than GRBG, need padding or cropping

    :param demosaic_net:
        demosaic-net

    :param device:
        'cuda:0', e.g.

    :return:
        rgb_hat:
          [batch_size, 3, m,n]  the rgb image predicted by the demosaic-net using our bayer input
    '''

    assert (cfa == 'GBRG') or (cfa == 'RGGB') or (cfa == 'GRBG') or (cfa == 'BGGR') 
    # 'only GBRG, RGGB, BGGR, GRBG are supported so far!'

    # print(im.shape)

    n_channel = im.shape[1]

    if n_channel==1:            # gray scale image
        im= torch.cat((im, im, im), 1)

    if cfa == 'GBRG':       # the demosiac net is trained w/ GRBG
        im = pad_gbrg_2_grbg(im,device)
    elif cfa == 'RGGB':
        im = pad_rggb_2_grbg(im, device)
    elif cfa == 'BGGR':
        im = pad_bggr_2_grbg(im, device)

    im= bayer_mosaic_tensor(im,device)

    sample = {"mosaic": im}

    rgb_hat = demosaic_net(sample)

    if cfa == 'GBRG':
        # an extra row and col is padded on four sides of the bayer before using demosaic-net. Need to trim the padded rows and cols of demosaiced rgb
        rgb_hat = unpad_grbg_2_gbrg(rgb_hat)
    elif cfa == 'RGGB':
        rgb_hat = unpad_grbg_2_rggb(rgb_hat)
    elif cfa == 'BGGR':
        rgb_hat = unpad_grbg_2_bggr(rgb_hat)

    rgb_hat = torch.clamp(rgb_hat, min=0, max=1)

    return rgb_hat


def pad_bggr_2_grbg(bayer, device):
    '''
            pad bggr bayer pattern to get grbg (for demosaic-net)

        :param bayer:
            2d tensor [bsz,ch, h,w]
        :param device:
            'cuda:0' or 'cpu', or ...
        :return:
            bayer: 2d tensor [bsz,ch,h,w+2]

        '''
    bsz, ch, h, w = bayer.shape

    bayer2 = torch.zeros([bsz, ch, h + 2, w], dtype=torch.float32)
    bayer2 = bayer2.to(device)

    bayer2[:, :, 1:-1, :] = bayer

    bayer2[:, :,  0, :] = bayer[:, :, 1, :]
    bayer2[:, :, -1, :] = bayer2[:, :, -2, :]

    bayer = bayer2

    return bayer


def pad_rggb_2_grbg(bayer,device):
    '''
        pad rggb bayer pattern to get grbg (for demosaic-net)

    :param bayer:
        2d tensor [bsz,ch, h,w]
    :param device:
        'cuda:0' or 'cpu', or ...
    :return:
        bayer: 2d tensor [bsz,ch,h,w+2]

    '''
    bsz, ch, h, w = bayer.shape

    bayer2 = torch.zeros([bsz,ch,h, w+2], dtype=torch.float32)
    bayer2 = bayer2.to(device)

    bayer2[:,:,:, 1:-1] = bayer

    bayer2[:,:,:, 0] =  bayer[:,:,:, 1]
    bayer2[:,:,:, -1] = bayer2[:,:,:, -2]

    bayer = bayer2

    return bayer


def pad_gbrg_2_grbg(bayer,device):
    '''
        pad gbrg bayer pattern to get grbg (for demosaic-net)

    :param bayer:
        2d tensor [bsz,ch, h,w]
    :param device:
        'cuda:0' or 'cpu', or ...
    :return:
        bayer: 2d tensor [bsz,ch,h+4,w+4]

    '''
    bsz, ch, h, w = bayer.shape

    bayer2 = torch.zeros([bsz,ch,h+2, w+2], dtype=torch.float32)
    bayer2 = bayer2.to(device)

    bayer2[:,:,1:-1, 1:-1] = bayer
    bayer2[:,:,0, 1:-1] = bayer[:,:,1, :]
    bayer2[:,:,-1, 1:-1] = bayer[:,:,-2, :]

    bayer2[:,:,:, 0] =  bayer2[:,:,:, 2]
    bayer2[:,:,:, -1] = bayer2[:,:,:, -3]

    bayer = bayer2

    return bayer


def unpad_grbg_2_gbrg(rgb):
    '''
        unpad the rgb image. this is used after pad_gbrg_2_grbg()
    :param rgb:
        tensor. [1,3,m,n]
    :return:
        tensor [1,3,m-2,n-2]

    '''
    rgb = rgb[:,:,1:-1,1:-1]

    return rgb


def unpad_grbg_2_bggr(rgb):
    '''
           unpad the rgb image. this is used after pad_bggr_2_grbg()
       :param rgb:
           tensor. [1,3,m,n]
       :return:
           tensor [1,3,m,n-2]

       '''
    rgb = rgb[:, :, 1:-1 , : ]

    return rgb


def unpad_grbg_2_rggb(rgb):
    '''
        unpad the rgb image. this is used after pad_rggb_2_grbg()
    :param rgb:
        tensor. [1,3,m,n]
    :return:
        tensor [1,3,m,n-2]

    '''
    rgb = rgb[:,:,:,1:-1]

    return rgb


def bayer_mosaic_tensor(im,device):
    '''
        create bayer mosaic to set as input to demosaic-net.
        make sure the input bayer (im) is GRBG.

    :param im:
            [batch_size, 3, m,n]. The color is in RGB order.
    :param device:
            'cuda:0', e.g.
    :return:
    '''

    """GRBG Bayer mosaic."""

    batch_size=im.shape[0]
    hh=im.shape[2]
    ww=im.shape[3]

    mask = torch.ones([batch_size,3,hh, ww], dtype=torch.float32)
    mask = mask.to(device)

    # red
    mask[:,0, ::2, 0::2] = 0
    mask[:,0, 1::2, :] = 0

    # green
    mask[:,1, ::2, 1::2] = 0
    mask[:,1, 1::2, ::2] = 0

    # blue
    mask[:,2, 0::2, :] = 0
    mask[:,2, 1::2, 1::2] = 0

    return im*mask