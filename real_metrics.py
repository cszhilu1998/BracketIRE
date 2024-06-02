# -*- coding: utf-8 -*-
import argparse
import glob
import os
from PIL import Image
from tqdm import tqdm
import torch
import sys
import cv2
import numpy as np
from collections import OrderedDict
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.registry import ARCH_REGISTRY


class InferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(
            self,
            metric_name,
            as_loss=False,
            loss_weight=None,
            loss_reduction='mean',
            device=None,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name

        # ============ set metric properties ===========
        # self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        if self.metric_mode is None:
            self.metric_mode = kwargs.pop('metric_mode')
        elif 'metric_mode' in kwargs:
            kwargs.pop('metric_mode')

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction

        # =========== define metric model ===============
        net_opts = OrderedDict()
        # load default setting first
        if metric_name in DEFAULT_CONFIGS.keys():
            default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
            net_opts.update(default_opt)
        # then update with custom setting
        net_opts.update(kwargs)
        network_type = net_opts.pop('type')
        self.net = ARCH_REGISTRY.get(network_type)(**net_opts)
        self.net = self.net.to(self.device)
        self.net.eval()

    def to(self, device):
        self.net.to(device)
        self.device = torch.device(device)
        return self

    def forward(self, target, ref=None, **kwargs):
        with torch.set_grad_enabled(self.as_loss):
            if 'afadaf' in self.metric_name:
                output = self.net(target, ref, device=self.device, **kwargs)
            else:
                if not torch.is_tensor(target):
                    target = imread2tensor(target)
                    target = target.unsqueeze(0)
                    if self.metric_mode == 'FR':
                        assert ref is not None, 'Please specify reference image for Full Reference metric'
                        ref = imread2tensor(ref)
                        ref = ref.unsqueeze(0)
                if self.metric_mode == 'FR':
                    output = self.net(target.to(self.device), ref.to(self.device), **kwargs)
                elif self.metric_mode == 'NR':
                    output = self.net(target.to(self.device), **kwargs)
        return output


def imread2tensor(img):
    img_tensor = torch.from_numpy(np.float32(img).transpose(2, 0, 1) / 65535.)
    return img_tensor


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="",
                        help='Name of the folder to save models and logs.')	
    parser.add_argument('--save_img', type=str, default="output_real_vispng_10")	
    parser.add_argument('--device', default="0")
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")

    # set up IQA model
    iqa_model_clip = InferenceModel(metric_name='clipiqa', metric_mode='NR', device=device)
    iqa_model_maniqa = InferenceModel(metric_name='maniqa', metric_mode='NR', device=device)

    root = sys.path[0]
    input_file = f'{root}/ckpt/{args.name}/{args.save_img}'
    epoch = args.save_img.split('_')[-1]
    save_file = f'{root}/ckpt/{args.name}/real_metrics_{epoch}.txt'

    print(f'Testing File: {input_file}')

    if os.path.isfile(input_file):
        input_paths = [input_file]
    else:
        input_dir = os.path.join(input_file, '**', '*.png')
        input_paths = sorted(glob.glob(input_dir, recursive = True))

    sf = open(save_file, 'a')
    sf.write(f'input address:\t{input_file}\n')
    p = sf.tell()

    avg_score_clip = 0
    avg_score_maniqa = 0
    
    test_img_num = len(input_paths)
    tqdm_input_paths = tqdm(input_paths)

    for idx, img_path in enumerate(tqdm_input_paths):
        img_name = os.path.basename(img_path)
        tar_img = cv2.imread(img_path, -1)[..., ::-1]
        H, W, C = tar_img.shape

        ref_img = None

        pre_img_clip = 0
        pre_img_maniqa = 0

        for i in range(16):
            if i==0:
                img = tar_img[0:H//4, 0:W//4].copy()
            elif i==1:
                img = tar_img[0:H//4, W//4:W//2].copy()
            elif i==2:
                img = tar_img[0:H//4, W//2:3*W//4].copy()
            elif i==3:
                img = tar_img[0:H//4, 3*W//4:W].copy()
            elif i==4:
                img = tar_img[H//4:H//2, 0:W//4].copy()
            elif i==5:
                img = tar_img[H//4:H//2, W//4:W//2].copy()
            elif i==6:
                img = tar_img[H//4:H//2, W//2:3*W//4].copy()
            elif i==7:
                img = tar_img[H//4:H//2, 3*W//4:W].copy()
            elif i==8:
                img = tar_img[H//2:3*H//4, 0:W//4].copy()
            elif i==9:
                img = tar_img[H//2:3*H//4, W//4:W//2].copy()
            elif i==10:
                img = tar_img[H//2:3*H//4, W//2:3*W//4].copy()
            elif i==11:
                img = tar_img[H//2:3*H//4, 3*W//4:W].copy()
            elif i==12:
                img = tar_img[3*H//4:H, 0:W//4].copy()
            elif i==13:
                img = tar_img[3*H//4:H, W//4:W//2].copy()
            elif i==14:
                img = tar_img[3*H//4:H, W//2:3*W//4].copy()
            elif i==15:
                img = tar_img[3*H//4:H, 3*W//4:W].copy()

            score_maniqa = iqa_model_maniqa(img, ref_img)
            pre_img_maniqa += score_maniqa
            torch.cuda.empty_cache()

            score_clip = iqa_model_clip(img, ref_img)
            pre_img_clip += score_clip
            torch.cuda.empty_cache()

        avg_score_clip += pre_img_clip / 16.
        avg_score_maniqa += pre_img_maniqa / 16.
        
        # print('%s  \t clipiqa: %.4f, \t musiq: %.3f, \t maniqa: %.4f , \t clipvit: %.4f  \n' % 
        #          (img_name, pre_img_clip / 16., pre_img_mus / 16., pre_img_maniqa / 16., pre_img_clipvit / 16.))
        sf.write('%s  \t clipiqa: %.4f, \t maniqa: %.4f  \n' % 
                 (img_name, pre_img_clip / 16., pre_img_maniqa / 16.))

    avg_score_clip /= test_img_num
    avg_score_maniqa /= test_img_num

    print('Average clipiqa score with %s images is: %.4f \n' % (test_img_num, avg_score_clip))
    print('Average maniqa score with %s images is: %.4f \n' % (test_img_num, avg_score_maniqa))

    sf.seek(p)
    sf.write('Average clipiqa score with %s images is: %.4f \n' % (test_img_num, avg_score_clip))
    sf.write('Average maniqa score with %s images is: %.4f \n' % (test_img_num, avg_score_maniqa))
    sf.close()

if __name__ == '__main__':
    with torch.no_grad():
        main()
