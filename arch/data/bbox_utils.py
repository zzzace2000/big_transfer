from torchvision.datasets import ImageFolder
import torchvision as tv
import os
import numpy as np
from torchvision.transforms import functional as F
from collections import namedtuple
import random
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
import copy
from torch.utils.data.sampler import Sampler
import bisect
from torch.nn.utils.rnn import pad_sequence


def bbox_collate(batch):
    '''
    Padding the bounding box in the collate function
    '''
    if torch.is_tensor(batch[0][0]):
        return default_collate(batch)
    if isinstance(batch[0][0], dict) and 'xs' not in batch[0][0]:
        return default_collate(batch)
    if batch[0][0].xs.ndim == 0:
        return default_collate(batch)

    samples = [item[0] for item in batch]
    # pad the bboxes into xs, ys, ws, hs
    data = {
        'xs': pad_sequence([sample.xs for sample in samples], batch_first=True, padding_value=-1.),
        'ys': pad_sequence([sample.ys for sample in samples], batch_first=True, padding_value=-1.),
        'ws': pad_sequence([sample.ws for sample in samples], batch_first=True, padding_value=-1.),
        'hs': pad_sequence([sample.hs for sample in samples], batch_first=True, padding_value=-1.),
    }

    data['imgs'] = default_collate([item['imgs'] for item in samples])
    if 'imgs_cf' in samples[0]:
        data['imgs_cf'] = default_collate([item['imgs_cf'] for item in samples])

    targets = default_collate([item[1] for item in batch])
    return [data, targets]


##################################################################
###############      BBox transformations       ##################
##################################################################
class RandomCrop(tv.transforms.RandomCrop):
    def __call__(self, sample):
        img = sample.imgs

        if 'xs' not in sample or (sample.xs < 0.).all():
            i, j, h, w = self.get_params(img, self.size)
            sample.imgs = F.crop(img, i, j, h, w)
            return sample

        low_i = (sample.ys - self.size[0] + 1).clamp_(0).min().item()
        low_j = (sample.xs - self.size[1] + 1).clamp_(0).min().item()
        max_i = (sample.ys + sample.hs - 1).max().item()
        max_j = (sample.xs + sample.ws - 1).max().item()

        # It has to contain at least 1 bounding box!
        while True:
            i, j, h, w = self.get_params(img, self.size, low_i=low_i, low_j=low_j,
                                         max_i=max_i, max_j=max_j)
            if 'xs' not in sample or (sample.xs < 0.).all():
                sample.imgs = F.crop(img, i, j, h, w)
                return sample

            new_xs = torch.clamp(sample.xs - j, min=0)
            new_ys = torch.clamp(sample.ys - i, min=0)
            new_ws = torch.min(
                ((sample.ws + sample.xs) - j).clamp_(min=0).sub_(new_xs),
                (w - new_xs))
            new_hs = torch.min(
                ((sample.hs + sample.ys) - i).clamp_(min=0).sub_(new_ys),
                (h - new_ys))

            # At least 1 bounding box is included
            if torch.any((new_ws != 0) & (new_hs != 0)):
                break
            else:
                print('Not found at least 1 valid bbox. Re-crop.')

        sample.xs = new_xs
        sample.ys = new_ys
        sample.ws = new_ws
        sample.hs = new_hs
        sample.imgs = F.crop(img, i, j, h, w)
        return sample

    @staticmethod
    def get_params(img, output_size, low_i=0, low_j=0, max_i=np.inf, max_j=np.inf):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.width, img.height
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(low_i, min(max_i, h - th))
        j = random.randint(low_j, min(max_j, w - tw))
        return i, j, th, tw


class CenterCrop(tv.transforms.CenterCrop):
    def __call__(self, sample):
        sample.imgs = F.center_crop(sample.imgs, self.size)
        if 'xs' not in sample or (sample.xs < 0.).all():
            return sample

        image_width, image_height = sample.imgs.size
        h, w = self.size
        j = int(round((image_height - h) / 2.))
        i = int(round((image_width - w) / 2.))

        new_xs = torch.clamp(sample.xs - j, min=0)
        new_ys = torch.clamp(sample.ys - i, min=0)
        new_ws = torch.min(
            ((sample.ws + sample.xs) - j).clamp_(min=0).sub_(new_xs),
            (w - new_xs))
        new_hs = torch.min(
            ((sample.hs + sample.ys) - i).clamp_(min=0).sub_(new_ys),
            (h - new_ys))

        sample.xs = new_xs
        sample.ys = new_ys
        sample.ws = new_ws
        sample.hs = new_hs
        return sample


class Resize(tv.transforms.Resize):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """
    def __call__(self, sample):
        h, w = sample.imgs.height, sample.imgs.width
        sample.imgs = F.resize(sample.imgs, self.size, self.interpolation)
        if 'xs' not in sample or (sample.xs < 0.).all():
            return sample

        new_h, new_w = sample.imgs.height, sample.imgs.width
        sample.xs, sample.ys, sample.ws, sample.hs = self.resize_bbox(
            w, h, new_w, new_h,
            sample.xs, sample.ys, sample.ws, sample.hs)

        # has_bbox = (sample.xs >= 0)
        # new_h, new_w = sample.imgs.height, sample.imgs.width
        #
        # old_xs, old_ys = sample.xs.clone(), sample.ys.clone()
        # sample.xs.mul_(new_w).floor_divide_(w)
        # sample.ys.mul_(new_h).floor_divide_(h)
        # # To be exact for w and h, we calculate the post-coordinate
        # # and round the coordinate to get width / height.
        # sample.ws.add_(old_xs).mul_(new_w).floor_divide_(w).add_(1).sub_(sample.xs)
        # sample.hs.add_(old_ys).mul_(new_h).floor_divide_(h).add_(1).sub_(sample.ys)

        # sample.xs[~has_bbox] = -1
        return sample

    @staticmethod
    def resize_bbox(w, h, new_w, new_h, bbox_x, bbox_y, bbox_w, bbox_h):
        new_bbox_x = bbox_x.mul(new_w).floor_divide_(w)
        new_bbox_y = bbox_y.mul(new_h).floor_divide_(h)

        new_bbox_w = bbox_w.add(bbox_x).mul_(new_w).floor_divide_(w).add_(1).sub_(new_bbox_x)
        new_bbox_h = bbox_h.add(bbox_y).mul_(new_h).floor_divide_(h).add_(1).sub_(new_bbox_y)

        return new_bbox_x, new_bbox_y, new_bbox_w, new_bbox_h


class RandomHorizontalFlip(tv.transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        if random.random() >= self.p:
            return sample

        sample.imgs = F.hflip(sample.imgs)
        if 'xs' not in sample or (sample.xs < 0.).all():
            return sample

        h, w = sample.imgs.height, sample.imgs.width
        sample.xs.add_(sample.ws).neg_().add_(w)
        return sample


class ColorJitter(tv.transforms.ColorJitter):
    def __call__(self, sample):
        sample.imgs = super().__call__(sample.imgs)
        return sample


class ToTensor(tv.transforms.ToTensor):
    def __call__(self, sample):
        sample.imgs = super().__call__(sample.imgs)
        return sample


class Normalize(tv.transforms.Normalize):
    def __call__(self, sample):
        sample.imgs = super().__call__(sample.imgs)
        return sample


class Grayscale(tv.transforms.Grayscale):
    def __call__(self, sample):
        sample.imgs = super().__call__(sample.imgs)
        return sample


# NIH dataset loader!!!
# https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L867
