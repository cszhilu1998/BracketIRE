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
import imageio


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
	LPIPS = 0.0 # calc_lpips(out, target, loss_fn_alex, range=ran)
	return np.array([psnr, SSIM, LPIPS], dtype=float)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Metrics for argparse')
	parser.add_argument('--dataroot', type=str, default='/home/data/Val_GT/')
	parser.add_argument('--device', default="0")
	parser.add_argument('--name', type=str, default="bracketire")
	parser.add_argument('--save_img', type=str, default="output_vispng_400")	
	args = parser.parse_args()

	device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
	loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').to(device)

	root = sys.path[0]
	save_file = f'{root}/ckpt/{args.name}/{args.save_img}/'
	files = [save_file]
	
	# files = [
	# 	root + '/ckpt/bracketire/output_vispng_400/',
	# ]
	
	for file in files:
		print('Start to measure images in %s...' % (file))
		metrics = np.zeros([len(os.listdir(file)), 3])
		log_dir = '%s/metrics_%s.txt' % (file.replace('/' + file.split('/')[-2] + '/', ''), file[-4:-1])
		print(log_dir)
		f = open(log_dir, 'a')
		i = 0

		for scene_file in tqdm(os.listdir(args.dataroot)):
			gt = cv2.imread(args.dataroot + scene_file, -1)[..., ::-1] 
			output = cv2.imread(file + scene_file, -1)[..., ::-1]

			# gt = imageio.imread(args.dataroot + scene_file, 'PNG-FI')
			# output = imageio.imread(file + scene_file, 'PNG-FI')

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
	
