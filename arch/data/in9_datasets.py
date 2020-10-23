


from torchvision.datasets import ImageFolder
import os
from os.path import join as pjoin, exists as pexists
import numpy as np
import torch
import random
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
import bisect
from torch.nn.utils.rnn import pad_sequence
from . import bbox_utils
import torchvision as tv
from PIL import Image


class IN9Dataset(ImageFolder):
    ''' Custom loader that loads images with bounding box '''

    def __init__(self, train_dir='../datasets/bg_challenge/train/original/train/',
                 no_fg_dir='../datasets/bg_challenge/train/no_fg/train/',
                 cf_inpaint_dir=None,
                 mask_only=False,
                 **kwargs):
        self.train_dir = train_dir
        self.no_fg_dir = no_fg_dir
        self.cf_inpaint_dir = cf_inpaint_dir

        if mask_only:
            mask_f = '../datasets/bg_challenge/train/original/have_mask_files'
            valid_files = set()
            with open(mask_f) as fp:
                for line in fp:
                    valid_files.add(line.strip())
            is_valid_file = lambda path: os.path.basename(path) in valid_files
            kwargs['is_valid_file'] = is_valid_file

        super().__init__(train_dir, **kwargs)

    def __getitem__(self, index):
        """
        Override this to return the bounding box as well
        """
        path, target = self.samples[index]
        imgs = self.loader(path)

        cls_name = self.classes[target]
        img_name = path.split("/")[-1]
        mask_path = pjoin(self.no_fg_dir, cls_name, img_name)

        sample = dict()
        sample['imgs'] = imgs
        if not pexists(mask_path): # no such mask exists
            sample['masks'] = None
        else:
            no_fg = self.loader(mask_path)
            is_zero = (np.asarray(no_fg) == 0.)
            is_diff = (np.asarray(no_fg) != np.asarray(imgs))
            is_mask = (is_zero & is_diff)
            if not is_mask.all(axis=2).any():
                sample['masks'] = None
            else:
                # extract x, y, w, h
                tmp = np.arange(is_mask.shape[0])[is_mask.all(axis=2).any(axis=1)]
                y, h = torch.tensor(tmp[0]), torch.tensor(tmp[-1] - tmp[0])
                tmp = np.arange(is_mask.shape[1])[is_mask.all(axis=2).any(axis=0)]
                x, w = torch.tensor(tmp[0]), torch.tensor(tmp[-1] - tmp[0])

                fg_mask = is_mask.astype('uint8')
                fg_mask = Image.fromarray(fg_mask * 255)

                sample.update({'masks': fg_mask, 'xs': x, 'ys': y, 'ws': w, 'hs': h})

        if self.cf_inpaint_dir is not None:
            cf_path = pjoin(self.cf_inpaint_dir, cls_name, img_name)
            sample['imgs_cf'] = self.loader(cf_path) \
                if pexists(cf_path) and sample['masks'] is not None \
                else None

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    @classmethod
    def get_train_transform(self, test_run=False):
        precut, cut = 256, 224
        if test_run:
            precut, cut = 16, 14

        train_bbox_tx = tv.transforms.Compose([
            bbox_utils.Resize((precut, precut)),
            bbox_utils.RandomCrop((cut, cut)),
            bbox_utils.RandomHorizontalFlip(),
            bbox_utils.ColorJitter(),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return train_bbox_tx

    @classmethod
    def get_val_transform(self, test_run=False):
        cut = 14 if test_run else 224

        val_bbox_tx = tv.transforms.Compose([
            bbox_utils.Resize((cut, cut)),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return val_bbox_tx

    def make_loader(self, batch_size, shuffle, workers):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True, drop_last=False,
            collate_fn=bbox_utils.bbox_collate)

    @property
    def is_bbox_folder(self):
        return True
