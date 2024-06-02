import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import glob
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
from os.path import join as opj
from util.util import mu_tonemap, save_hdr
from isp.isp import isp_pip
import exifread
import cv2


def flatten_raw_image(im_raw_4ch):  # 4xHxW
    """ unpack a 4-channel tensor into a single channel bayer image"""
    if isinstance(im_raw_4ch, np.ndarray):
        im_out = np.zeros_like(im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2))
    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros((im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype)
    else:
        raise Exception

    im_out[0::2, 0::2] = im_raw_4ch[1, :, :]
    im_out[0::2, 1::2] = im_raw_4ch[0, :, :]
    im_out[1::2, 0::2] = im_raw_4ch[3, :, :]
    im_out[1::2, 1::2] = im_raw_4ch[2, :, :]

    return im_out  # HxW


if __name__ == '__main__':
    opt = TestOptions().parse()

    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, 'test', opt)
        datasets[dataset_name] = tqdm(dataset)

    for load_iter in load_iters:
        opt.load_iter = load_iter
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        log_dir = '%s/%s/logs/log_epoch_%d.txt' % (
                opt.checkpoints_dir, opt.name, load_iter)
        os.makedirs(os.path.split(log_dir)[0], exist_ok=True)
        f = open(log_dir, 'a')

        for dataset_name in dataset_names:
            opt.dataset_name = dataset_name
            tqdm_val = datasets[dataset_name]
            dataset_test = tqdm_val.iterable
            dataset_size_test = len(dataset_test)

            print('='*80)
            print(dataset_name + ' dataset')
            tqdm_val.reset()

            psnr = [0.0] * dataset_size_test

            time_val = 0
            for i, data in enumerate(tqdm_val):
                torch.cuda.empty_cache()
                model.set_input(data)
                torch.cuda.synchronize()
                time_val_start = time.time()
                model.test()
                torch.cuda.synchronize()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()

                if opt.save_imgs:
                    file_name = data['fname'][0]
                    
                    folder_dir = './ckpt/%s/output_real_vispng_%d' % (opt.name, load_iter)  
                    os.makedirs(folder_dir, exist_ok=True)
                    save_dir_vispng = '%s/%s.png' % (folder_dir, file_name)
                    raw_img = torch.clamp(res['data_out'][0] / 16, 0, 1)
                    raw_img = flatten_raw_image(raw_img)

                    raw_path = sorted(glob.glob(opj(opt.dataroot, 'Test', file_name, '*.dng')))[0]
                    exir_file = exifread.process_file(open(raw_path, 'rb'), details=False, strict=True)

                    dict={}
                    for key, value in exir_file.items():
                        dict[key] = value.values

                    img = isp_pip(raw_img.cpu().numpy(), dict, device=res['data_out'].device) 
                    img = mu_tonemap(img, mu=5000)
                    img = np.clip(img*65535, 0, 65535)
                    img = img[..., ::-1]
                    cv2.imwrite(save_dir_vispng, img.astype(np.uint16))
            
            avg_psnr = '%.2f'%np.mean(psnr)

            f.write('dataset: %s, PSNR: %s, Time: %.3f s, AVG Time: %.3f ms\n' 
                     % (dataset_name, avg_psnr, time_val, time_val/dataset_size_test*1000))
            print('Time: %.3f s AVG Time: %.3f ms PSNR: %s\n' % (time_val, time_val/dataset_size_test*1000, avg_psnr))
            f.flush()
            f.write('\n')
        f.close()
    for dataset in datasets:
        datasets[dataset].close()
