import glob
import os
import cv2
import numpy as np

from .pipeline import run_pipeline_v2
from .pipeline_utils import get_visible_raw_image, get_metadata

params = {
	'input_stage': 'normal',  # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
	'output_stage': 'srgb',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
	'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
	'demosaic_type': 'net', # 'menon2007', 'EA', 'VNG', 'net', 'down
	'save_dtype': np.uint16
}

def reshape_back_raw(bayer):
	H = bayer.shape[1]
	W = bayer.shape[2]
	newH = int(H*2)
	newW = int(W*2)
	bayer_back = np.zeros((newH, newW))
	bayer_back[0:newH:2, 0:newW:2] = bayer[3]
	bayer_back[0:newH:2, 1:newW:2] = bayer[1]
	bayer_back[1:newH:2, 0:newW:2] = bayer[2]
	bayer_back[1:newH:2, 1:newW:2] = bayer[0]
	
	return bayer_back

def isp_pip(raw_image, meta_data, device='0'):
	# raw_image = reshape_back_raw(npy_img) / ratio

	# metadata
	# meta_npy = meta_data.items()
	metadata = get_metadata(meta_data)
	# raw_image = raw_image * (2**10 - 1) # + meta_npy['black_level'][0]

	# render
	output_image = run_pipeline_v2(raw_image, params, metadata=metadata, device=device)
	# output_image = output_image[..., ::-1] * 255
	# output_image = np.clip(output_image, 0, 255)

	return output_image #.astype(params['save_dtype'])
