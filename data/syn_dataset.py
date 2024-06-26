import numpy as np
import os
import cv2
import torch
import random
from tqdm import tqdm
from os.path import join as opj
from multiprocessing.dummy import Pool
from data.base_dataset import BaseDataset


# BracketIRE dataset
class SynDataset(BaseDataset):
    def __init__(self, opt, split='train', dataset_name='BracketIRE'):
        super(SynDataset, self).__init__(opt, split, dataset_name)

        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size
        self.frame_num = opt.frame_num

        if split == 'train':
            self._getitem = self._getitem_train
            self.names, self.meta_dirs, self.raw_dirs, self.gt_dirs = self._get_image_dir(self.root, split, name='Train')
            self.len_data = 500 * self.batch_size
        elif split == 'val':
            self._getitem = self._getitem_val
            self.names, self.meta_dirs, self.raw_dirs, self.gt_dirs = self._get_image_dir(self.root, split, name='Test')
            self.len_data = len(self.names)
        elif split == 'test': 
            self._getitem = self._getitem_test
            self.names, self.meta_dirs, self.raw_dirs, self.gt_dirs = self._get_image_dir(self.root, split, name='Test')
            self.len_data = len(self.names)
        else:
            raise ValueError

        self.split = split
        self.meta_data = [0] * len(self.names)
        self.raw_images = [0] * len(self.names)
        self.gt_images = [0] * len(self.names)
        read_images(self)

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data

    def _getitem_train(self, idx):
        idx = idx % len(self.names)
        raws = torch.from_numpy(np.float32(np.array(self.raw_images[idx]))) / (2**10 - 1)
        gt = torch.from_numpy(np.float32(self.gt_images[idx]))
        raws, gt = self._crop_patch(raws, gt, self.patch_size)

        return {'gt': gt, # [4, H, W]
                'raws': raws, # [T=5, 4, H, W]
                'fname': self.names[idx]}

    def _getitem_val(self, idx):   
        raws = torch.from_numpy(np.float32(np.array(self.raw_images[idx]))) / (2**10 - 1)
        gt = torch.from_numpy(np.float32(self.gt_images[idx]))
        meta = self._process_metadata(self.meta_data[idx])

        return {'meta': meta,
                'gt': gt[..., 10:10+128, 10:10+512],
                'raws': raws[..., 10:10+128, 10:10+512],
                'fname': self.names[idx]}

    def _getitem_test(self, idx):     
        raws = torch.from_numpy(np.float32(np.array(self.raw_images[idx]))) / (2**10 - 1)
        gt = torch.from_numpy(np.float32(self.gt_images[idx]))
        meta = self._process_metadata(self.meta_data[idx])

        return {'meta': meta, 
                'gt': gt,
                'raws': raws[..., 5:-5, 5:-5],
                'fname': self.names[idx]}

    def _crop_patch(self, raws, gt, p):
        ih, iw = raws.shape[-2:]
        ph = random.randrange(10, ih - p + 1 - 10)
        pw = random.randrange(10, iw - p + 1 - 10)
        return raws[..., ph:ph+p, pw:pw+p], \
               gt[..., ph:ph+p, pw:pw+p]

    def _process_metadata(self, metadata):
        metadata_item = metadata.item()
        meta = {}
        for key in metadata_item:
            meta[key] = torch.from_numpy(metadata_item[key])
        return meta

    def _read_raw_path(self, root):
        img_paths = []
        for expo in range(self.frame_num):
            img_paths.append(opj(root, 'raw_' + str(4**expo) + '.npy'))
        return img_paths

    def _get_image_dir(self, dataroot, split=None, name=None):
        image_names = []
        meta_dirs = []
        raw_dirs = []
        gt_dirs = []

        if split == 'train' or split == 'test':
            for scene_file in sorted(os.listdir(opj(dataroot, name))): 
                for image_file in sorted(os.listdir(opj(dataroot, name, scene_file))): 
                    image_root = opj(dataroot, name, scene_file, image_file)
                    image_names.append(scene_file + '-' + image_file)
                    meta_dirs.append(opj(image_root, 'metadata.npy'))
                    raw_dirs.append(self._read_raw_path(image_root))
                    gt_dirs.append(opj(image_root, 'raw_gt.npy'))
        elif split == 'val':
            for scene_file in sorted(os.listdir(opj(dataroot, name))): 
                for image_file in sorted(os.listdir(opj(dataroot, name, scene_file)))[0::20]: 
                    image_root = opj(dataroot, name, scene_file, image_file)
                    image_names.append(scene_file + '-' + image_file)
                    meta_dirs.append(opj(image_root, 'metadata.npy'))
                    raw_dirs.append(self._read_raw_path(image_root))
                    gt_dirs.append(opj(image_root, 'raw_gt.npy'))
        else:
            raise ValueError
        
        return image_names, meta_dirs, raw_dirs, gt_dirs


def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    for _ in range(3):
        try:
            imgs = []
            for m in range(obj.frame_num):
                 imgs.append(np.load(obj.raw_dirs[i][m], allow_pickle=True).transpose(2, 0, 1))
            obj.raw_images[i] = imgs
            obj.gt_images[i] = np.load(obj.gt_dirs[i], allow_pickle=True).transpose(2, 0, 1)
            obj.meta_data[i] = np.load(obj.meta_dirs[i], allow_pickle=True)
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
