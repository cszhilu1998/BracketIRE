import numpy as np
import cv2
import torch
import math
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim
import argparse
import sys
import os
from options.base_options import str2bool


def calc_psnr_np(sr, hr, range):
    diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
    mse = np.power(diff, 2).mean()
    return -10 * math.log10(mse)

def lpips_norm(img, range):
    img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
    img = img / (range / 2.) - 1
    return torch.Tensor(img).to(device)

def calc_lpips(out, target, loss_fn_alex, range):
    lpips_out = lpips_norm(out, range)
    lpips_target = lpips_norm(target, range)
    LPIPS = loss_fn_alex(lpips_out, lpips_target)
    return LPIPS.detach().cpu().item()

def calc_metrics(out, target, loss_fn_alex):
    ran = 65535.0
    psnr = calc_psnr_np(out, target, range=ran)
    SSIM = ssim(out, target, win_size=11, data_range=ran, channel_axis=2, gaussian_weights=True)
    LPIPS = calc_lpips(out, target, loss_fn_alex, range=ran)
    return np.array([psnr, SSIM, LPIPS], dtype=float)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Metrics for argparse')
    parser.add_argument('--dataroot', type=str, default='/Data/dataset/MultiExpo/Syn/')
    parser.add_argument('--name', type=str, default="bracketire")
    parser.add_argument('--plus', type=str2bool, default=False)
    parser.add_argument('--save_img', type=str, default="output_syn_vispng_400")	
    parser.add_argument('--device', default="0")
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').to(device)

    root = sys.path[0]
    save_file = f'{root}/ckpt/{args.name}/{args.save_img}/'
    files = [save_file]
    
    # files = [
    #     root + '/ckpt/tmrnet/output_vispng_400/',
    # ]

    for file in files:
        print('Start to measure images in %s...' % (file))
        metrics = np.zeros([290, 3])
        log_dir = '%s/syn_metrics_%s.txt' % (file.replace('/' + file.split('/')[-2] + '/', ''), file[-4:-1])
        print(log_dir)
        f = open(log_dir, 'a')
        i = 0

        for scene_file in os.listdir(args.dataroot + 'Test/'):
            for image_file in tqdm(os.listdir(args.dataroot + 'Test/' + scene_file + '/')): 
                gt = cv2.imread(args.dataroot + 'Test/' + scene_file + '/' + image_file + '/rgb_vis_gt.png', -1)[..., ::-1] 
                output = cv2.imread(file + scene_file + '/' + image_file + '.png', -1)[..., ::-1]

                # Crop surrounding pixels
                if not args.plus:
                    gt = gt[10:-10, 10:-10]
                    output = output[10:-10, 10:-10]
                else:
                    gt = gt[16:-16, 16:-16]
                    output = output[16:-16, 16:-16]

                metrics[i, 0:3] = calc_metrics(output, gt, loss_fn_alex_v1)
                i = i + 1

        mean_metrics = np.mean(metrics, axis=0)
    
        print('\n        File        :\t %s \n' % (file))
        print('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                % (mean_metrics[0], mean_metrics[1], mean_metrics[2]))
        f.write('\n        File        :\t %s \n' % (file))
        f.write('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
                % (mean_metrics[0], mean_metrics[1], mean_metrics[2]))
        f.flush()
        f.close()