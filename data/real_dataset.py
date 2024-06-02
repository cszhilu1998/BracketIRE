import numpy as np
import os
import glob
import torch
import random
import rawpy
from tqdm import tqdm
from os.path import join as opj
from multiprocessing.dummy import Pool
from data.base_dataset import BaseDataset


# RealDataset dataset
class RealDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='Real'):
        super(RealDataset, self).__init__(opt, split, dataset_name)

        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size 
        self.frame_num = opt.frame_num

        self.names, self.raw_dirs, self.expo_dirs = self._get_image_dir(self.root, split)

        if split == 'train':
            self._getitem = self._getitem_train
            self.len_data = 500 * self.batch_size
        elif split == 'val':
            self._getitem = self._getitem_val
            self.len_data = len(self.names)
        elif split == 'test':
            self._getitem = self._getitem_test
            self.len_data = len(self.names)
        else:
            raise ValueError
        
        self.raw_images = [0] * len(self.names)
        self.expos = [0] * len(self.names)
        read_images(self)

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data

    def _getitem_train(self, idx):     
        idx = idx % len(self.names)
        raws = self.raw_images[idx]

        raws = self._crop_patch(raws, self.patch_size)
        raws = (np.float32(raws) - 64) / (2**10 - 1 - 64)
        raws = torch.from_numpy(np.clip(raws, 0, 1))

        expos = torch.from_numpy(self.expos[idx][2] / self.expos[idx])

        return {'raws': raws,
                'gt': raws[2],
                'expos': expos,
                'fname': self.names[idx]}

    def _getitem_val(self, idx):     
        raws = self.raw_images[idx]

        raws = (np.float32(raws) - 64) / (2**10 - 1 - 64)
        raws = torch.from_numpy(np.clip(raws, 0, 1))

        expos = torch.from_numpy(self.expos[idx][2] / self.expos[idx])

        return {'raws': raws[..., 1000:1128, 1000:1512],
                'expos': expos,
                'gt': raws[2],
                'fname': self.names[idx]}

    def _getitem_test(self, idx):     
        raws = self.raw_images[idx]

        raws = (np.float32(raws) - 64) / (2**10 - 1 - 64)
        raws = torch.from_numpy(np.clip(raws, 0, 1))

        expos = torch.from_numpy(self.expos[idx][2] / self.expos[idx])

        return {'raws': raws,
                'expos': expos,
                'gt': raws[2],
                'fname': self.names[idx]}

    def _crop_patch(self, raws, p):
        ih, iw = raws.shape[-2:]
        ph = random.randrange(0, ih - p + 1)
        pw = random.randrange(0, iw - p + 1)
        return raws[..., ph:ph+p, pw:pw+p]

    def _get_image_dir(self, dataroot, split=None):
        image_names = []
        raw_dirs = []
        expo_dirs = []

        if split=='train':
            for scene_file in sorted(os.listdir(opj(dataroot, 'Train'))): 
                ims_paths = sorted(glob.glob(opj(dataroot, 'Train', scene_file, '*.dng')))
                image_names.append(scene_file)
                raw_dirs.append(ims_paths)
                expo_dirs.append(opj(dataroot, 'Train', scene_file, 'exposure.txt'))
        elif split=='val':
            for scene_file in sorted(os.listdir(opj(dataroot, 'Test')))[0::3]: 
                ims_paths = sorted(glob.glob(opj(dataroot, 'Test', scene_file, '*.dng')))
                image_names.append(scene_file)
                raw_dirs.append(ims_paths)
                expo_dirs.append(opj(dataroot, 'Test', scene_file, 'exposure.txt'))
        elif split=='test':
            for scene_file in sorted(os.listdir(opj(dataroot, 'Test'))): 
                ims_paths = sorted(glob.glob(opj(dataroot, 'Test', scene_file, '*.dng')))
                image_names.append(scene_file)
                raw_dirs.append(ims_paths)
                expo_dirs.append(opj(dataroot, 'Test', scene_file, 'exposure.txt'))
        return image_names, raw_dirs, expo_dirs

def pack_raw_image(im_raw):  # HxW
    """ Packs a single channel bayer image into 4 channel tensor, where channels contain R, G, G, and B values"""
    if isinstance(im_raw, np.ndarray):
        im_out = np.zeros_like(im_raw, shape=(im_raw.shape[0], 4, im_raw.shape[1] // 2, im_raw.shape[2] // 2))
    elif isinstance(im_raw, torch.Tensor):
        im_out = torch.zeros((im_raw.shape[0], 4, im_raw.shape[1] // 2, im_raw.shape[2] // 2), dtype=im_raw.dtype)
    else:
        raise Exception

    im_out[:, 2, :, :] = im_raw[:, 1::2, 1::2]
    im_out[:, 0, :, :] = im_raw[:, 0::2, 1::2]
    im_out[:, 3, :, :] = im_raw[:, 1::2, 0::2]
    im_out[:, 1, :, :] = im_raw[:, 0::2, 0::2]
    
    return im_out  # 4xHxW
    
def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    for _ in range(3):
        try:
            imgs = []
            for m in range(obj.frame_num):
                imgs.append(rawpy.imread(obj.raw_dirs[i][m]).raw_image_visible)
            obj.raw_images[i] = pack_raw_image(np.array(imgs))
            obj.expos[i] = np.float32(np.loadtxt(obj.expo_dirs[i]))
            failed = False
            break
        except:
            failed = True
    if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
    # may use `from multiprocessing import Pool` instead, but less efficient and
    # NOTE: `multiprocessing.Pool` will duplicate given object for each process.
    print('Starting to load images via multiple imreaders')
    pool = Pool() # use all threads by default
    for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
        pass
    pool.close()
    pool.join()

if __name__ == '__main__':
    pass
