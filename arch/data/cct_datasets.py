'''
The loader for Caltech Camera Traps (CCT) datasets.
https://beerys.github.io/CaltechCameraTraps/
'''

import os

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from torchvision.datasets.folder import default_loader
import json
from os.path import join as pjoin, exists as pexists

from . import bbox_utils
from ..utils import DotDict

thispath = os.path.dirname(os.path.realpath(__file__))


class MyCCT_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 json_file='../datasets/cct/eccv_18_annotation_files/train_annotations.json',
                 cct_img_folder='../datasets/cct/eccv_18_all_images_sm/',
                 transform=None):
        super().__init__()
        self.json_file = json_file
        self.cct_img_folder = cct_img_folder
        self.transform = transform

        # Load json
        self.setup_data()

    def setup_data(self):
        tmp = json.load(open(self.json_file))
        # It has 'image_id', 'category_id', 'bbox', 'id'
        annotations = pd.DataFrame(tmp['annotations'])
        images = pd.DataFrame(tmp['images'])[['id', 'height', 'width']]

        self.annotations = pd.merge(annotations, images,
                 how='left', left_on=['image_id'], right_on=['id'])
        self.annotations = self.annotations[[
            'image_id', 'category_id', 'bbox', 'height', 'width']]

        # setup category_id to the target id
        tmp2 = pd.DataFrame(tmp['categories']).sort_values('id').reset_index(drop=True)
        self.y_to_cat_id = tmp2['id']
        self.cat_id_to_y = pd.Series(self.y_to_cat_id.index,
                                     self.y_to_cat_id.values)
        self.category_name = tmp2['name']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        record = self.annotations.loc[idx]
        img_path = pjoin(self.cct_img_folder, record['image_id'] + '.jpg')
        img = default_loader(img_path)

        target = self.cat_id_to_y[record.category_id]

        result = DotDict()
        result['imgs'] = img
        if np.isnan(record.bbox).any():
            result['xs'] = result['ys'] = result['ws'] = result['hs'] = \
                torch.tensor(-1)
        else:
            w, h = record.width, record.height
            new_w, new_h = img.width, img.height
            bbox = [torch.tensor(int(v)) for v in record.bbox]
            # Do transformation!!!
            if w != new_w or h != new_h:
                bbox = bbox_utils.Resize.resize_bbox(
                    w, h, new_w, new_h, *bbox)

            result['xs'], result['ys'], result['ws'], result['hs'] \
                = bbox

        if self.transform is not None:
            result = self.transform(result)
        return result, target

    @property
    def is_bbox_folder(self):
        return True

    @classmethod
    def get_train_bbox_transform(cls, test_run=False):
        orig_size = 224 if not test_run else 56

        train_tx = tv.transforms.Compose([
            bbox_utils.Resize(orig_size),
            bbox_utils.RandomCrop((orig_size, orig_size)),
            bbox_utils.RandomHorizontalFlip(),
            bbox_utils.ColorJitter(),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5,), (0.5,)),
        ])
        return train_tx

    @classmethod
    def get_val_bbox_transform(cls, test_run=False):
        orig_size = 224 if not test_run else 56
        val_tx = tv.transforms.Compose([
            bbox_utils.Resize((orig_size, orig_size)),
            bbox_utils.ToTensor(),
            bbox_utils.Normalize((0.5,), (0.5,)),
        ])
        return val_tx

    def make_loader(self, batch_size, shuffle, workers):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True, drop_last=False,
            collate_fn=bbox_utils.bbox_collate)
