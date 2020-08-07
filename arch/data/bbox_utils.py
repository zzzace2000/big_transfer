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


##################################################################
###############      BBox transformations       ##################
##################################################################
class RandomCrop(tv.transforms.RandomCrop):
    def __call__(self, sample):
        img = sample.imgs

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        low_i = (sample.ys - self.size[0]).clamp_(0).min().item()
        low_j = (sample.xs - self.size[1]).clamp_(0).min().item()

        # It has to contain at least 1 bounding box!
        while True:
            i, j, h, w = self.get_params(img, self.size, low_i=low_i, low_j=low_j)
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
    def get_params(img, output_size, low_i=0, low_j=0):
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

        i = random.randint(low_i, h - th)
        j = random.randint(low_j, w - tw)
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

        # has_bbox = (sample.xs >= 0)
        new_h, new_w = sample.imgs.height, sample.imgs.width

        old_xs, old_ys = sample.xs.clone(), sample.ys.clone()
        sample.xs.mul_(new_w).floor_divide_(w)
        sample.ys.mul_(new_h).floor_divide_(h)
        # To be exact for w and h, we calculate the post-coordinate
        # and round the coordinate to get width / height.
        sample.ws.add_(old_xs).mul_(new_w).floor_divide_(w).add_(1).sub_(sample.xs)
        sample.hs.add_(old_ys).mul_(new_h).floor_divide_(h).add_(1).sub_(sample.ys)

        # sample.xs[~has_bbox] = -1
        return sample


class RandomHorizontalFlip(tv.transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        if random.random() >= self.p:
            return sample

        sample.imgs = F.hflip(sample.imgs)
        if 'xs' not in sample or (sample.xs < 0.).all():
            return sample

        # has_bbox = (sample.xs >= 0)
        h, w = sample.imgs.height, sample.imgs.width
        sample.xs.add_(sample.bbox.ws).neg_().add_(w)
        # sample.xs[~has_bbox] = -1
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
