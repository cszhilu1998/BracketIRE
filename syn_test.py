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
from util.process import get_raw2rgb
from util.util import mu_tonemap
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
                    # path for saving images
                    file_name = data['fname'][0].split('-')
                    folder_dir = './ckpt/%s/output_syn_vispng_%03d/%s' % (opt.name, load_iter, file_name[0])  
                    os.makedirs(folder_dir, exist_ok=True)
                    save_dir_vispng = '%s/%s.png' % (folder_dir, file_name[1])

                    raw_img = res['data_out'][0].permute(1, 2, 0) / 16 
                    img = get_raw2rgb(raw_img, data['meta'], demosaic='net', lineRGB=True)
                    img = torch.clamp(mu_tonemap(img, mu=5e3)*65535, 0, 65535)
                    img = img.cpu().numpy()[..., ::-1]

                    # pad surrounding pixels with 0 values
                    if dataset_name == 'syn':
                        img = np.pad(img, ((10,10), (10,10), (0,0)), 'constant', constant_values=((0,0), (0,0), (0,0)))
                    elif dataset_name == 'synplus':
                        img = np.pad(img, ((16,16), (16,16), (0,0)), 'constant', constant_values=((0,0), (0,0), (0,0)))
                    else:
                        raise ValueError

                    cv2.imwrite(save_dir_vispng, img.astype(np.uint16))

            print('dataset: %s, Time: %.3f s, AVG Time: %.3f ms \n' 
                  % (dataset_name, time_val, time_val/dataset_size_test*1000))
    
    for dataset in datasets:
        datasets[dataset].close()
