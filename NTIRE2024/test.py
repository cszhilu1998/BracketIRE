import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from util.util import calc_psnr as calc_psnr
import time
import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
from data.degrade.degrade_kernel import get_raw2rgb
from data.degrade.process import gamma_compression
from util.util import mu_tonemap, save_hdr
import cv2


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

            folder_dir = './ckpt/%s/output_vispng_%d' % (opt.name, load_iter)  
            os.makedirs(folder_dir, exist_ok=True)

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
                    save_dir_vispng = '%s/%s.png' % (folder_dir,  data['fname'][0])
                    raw_img = res['data_out'][0].permute(1, 2, 0) / 16 
                    img = get_raw2rgb(raw_img, data['meta'], demosaic='net', lineRGB=True)
                    img = torch.clamp(mu_tonemap(img, mu=5e3)*65535, 0, 65535)
                    img = img.cpu().numpy()[..., ::-1]
                    cv2.imwrite(save_dir_vispng, img.astype(np.uint16))

            avg_psnr = '%.2f'%np.mean(psnr)

    for dataset in datasets:
        datasets[dataset].close()
