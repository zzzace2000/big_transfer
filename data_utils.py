from torchvision.datasets import ImageFolder
import torchvision as tv
import os
from torchvision.transforms import functional as F
from collections import namedtuple
import random
import torch
from torch.utils.data._utils.collate import default_collate
import copy


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, DotDict):
                self[k] = DotDict(v)

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo=memo))


BBox = namedtuple('BBox', 'xs ys ws hs')
Sample = namedtuple('Sample', 'img bbox')


def bbox_collate(batch):
    '''
    Storing a list of bbox to handle variable length.

    Annoyingly, the pin_memory in the loader would destroy dotdict
    and change it back to dict. Not sure how to solve this...
    '''
    samples = [item[0] for item in batch]
    imgs = default_collate([item.img for item in samples])

    bboxes = [item.bbox for item in samples]
    data = Sample(img=imgs, bbox=bboxes)

    targets = default_collate([item[1] for item in batch])
    return [data, targets]


class ImagenetBoundingBoxFolder(ImageFolder):
    ''' Custom loader that loads images with bounding box '''

    def __init__(self, root, bbox_file, only_imgs_with_bbox=True, **kwargs):
        ''' bbox_file points to either `LOC_train_solution.csv` or `LOC_val_solution.csv` '''
        self.coord_dict = self.parse_coord_dict(bbox_file)
        self.only_imgs_with_bbox = only_imgs_with_bbox

        if only_imgs_with_bbox:
            kwargs['is_valid_file'] = \
                lambda path: os.path.basename(path) in self.coord_dict

        super().__init__(root, **kwargs)

    @staticmethod
    def parse_coord_dict(data_file):
        # map from ILSVRC2012_val_00037956 to ('n03995372', [85 1 499 272])
        coord_dict = {}
        with open(data_file) as fp:
            fp.readline()
            for line in fp:
                line = line.strip().split(',')
                filename = '%s.JPEG' % line[0]
                tmp = line[1].split(' ')

                xs, ys, ws, hs = [], [], [], []
                the_first_class = tmp[0]
                for i in range(len(tmp) // 5):
                    the_class = tmp[i * 5]
                    if the_class != the_first_class:
                        continue

                    # The string is: n0133595 x1 y1 x2 y2
                    [x1, y1, x2, y2] = tmp[(i * 5 + 1):(i * 5 + 5)]

                    # parse it in x, y, w, h
                    xs.append(int(x1))
                    ys.append(int(y1))
                    ws.append((int(x2) - int(x1)))
                    hs.append((int(y2) - int(y1)))

                # Only take the first bounding box which is the ground truth
                import numpy as np
                coord_dict[filename] = DotDict(
                    xs=torch.LongTensor(xs),
                    ys=torch.LongTensor(ys),
                    ws=torch.LongTensor(ws),
                    hs=torch.LongTensor(hs),
                )

        return coord_dict

    def __getitem__(self, index):
        """
        Override this to return the bounding box as well
        """
        path, target = self.samples[index]
        img = self.loader(path)

        # Append the bounding box in the 4th channel
        filename = os.path.basename(path)

        bbox = self.coord_dict[filename] if filename in self.coord_dict else None
        sample = DotDict(img=img, bbox=DotDict(
            xs=bbox['xs'].clone(),
            ys=bbox['ys'].clone(),
            ws=bbox['ws'].clone(),
            hs=bbox['hs'].clone(),
        ))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class MyHackSampleSizeMixin(object):
    '''
    Custom dataset to hack the lightning framework.
    To train a fixed number of steps, since lightning only supports
    epoch-based training. So this hack is to return a dataset that
    has special length to make resulting loader run for an epoch.
    '''
    def __init__(self, root, num_samples=None, **kwargs):
        self.num_samples = num_samples
        super().__init__(root, **kwargs)

    def __len__(self):
        if self.num_samples is None:
            return super().__len__()

        return self.num_samples

    def __getitem__(self, index):
        if self.num_samples is None:
            return super().__getitem__(index)

        actual_len = super().__len__()
        return super().__getitem__(index % actual_len)

class MyImageFolder(MyHackSampleSizeMixin, ImageFolder):
    pass

class MyImagenetBoundingBoxFolder(MyHackSampleSizeMixin, ImagenetBoundingBoxFolder):
    pass

class RandomCrop(tv.transforms.RandomCrop):
    def __call__(self, sample):
        img, bbox = sample.img, sample.bbox

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        low_i = (bbox.ys - self.size[0]).clamp_(0).min().item()
        low_j = (bbox.xs - self.size[1]).clamp_(0).min().item()

        # It has to contain at least 1 bounding box!
        while True:
            i, j, h, w = self.get_params(img, self.size, low_i=low_i, low_j=low_j)
            if bbox is None:
                sample.img = F.crop(img, i, j, h, w)
                return sample

            new_xs = torch.clamp(bbox.xs - j, min=0)
            new_ys = torch.clamp(bbox.ys - i, min=0)
            new_ws = torch.min(
                ((bbox.ws + bbox.xs) - j).clamp_(min=0).sub_(new_xs),
                (w - new_xs))
            new_hs = torch.min(
                ((bbox.hs + bbox.ys) - i).clamp_(min=0).sub_(new_ys),
                (h - new_ys))

            # At least 1 bounding box is included
            if torch.any((new_ws != 0) & (new_hs != 0)):
                break
            else:
                print('Not found at least 1 valid bbox. Re-crop.')

        sample.bbox.xs = new_xs
        sample.bbox.ys = new_ys
        sample.bbox.ws = new_ws
        sample.bbox.hs = new_hs
        sample.img = F.crop(img, i, j, h, w)
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
        h, w = sample.img.height, sample.img.width
        sample.img = F.resize(sample.img, self.size, self.interpolation)
        if sample.bbox is None:
            return sample

        new_h, new_w = sample.img.height, sample.img.width

        old_xs, old_ys = sample.bbox.xs.clone(), sample.bbox.ys.clone()
        sample.bbox.xs.mul_(new_w).div_(w)
        sample.bbox.ys.mul_(new_h).div_(h)
        # To be exact for w and h, we calculate the post-coordinate
        # and round the coordinate to get width / height.
        # sample.bbox.ws.mul_(new_w).div_(w).add_(1)
        # sample.bbox.hs.mul_(new_h).div_(h).add_(1)
        sample.bbox.ws.add_(old_xs).mul_(new_w).div_(w).add_(1).sub_(sample.bbox.xs)
        sample.bbox.hs.add_(old_ys).mul_(new_h).div_(h).add_(1).sub_(sample.bbox.ys)
        return sample


class RandomHorizontalFlip(tv.transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        if random.random() >= self.p:
            return sample

        sample.img = F.hflip(sample.img)
        if sample.bbox is None:
            return sample

        h, w = sample.img.height, sample.img.width
        sample.bbox.xs.add_(sample.bbox.ws).neg_().add_(w)
        return sample


class ToTensor(tv.transforms.ToTensor):
    def __call__(self, sample):
        sample.img = F.to_tensor(sample.img)
        if sample.bbox is None:
            return sample

        # It seems that the bbox can't be a mutable object
        sample.bbox = BBox(**sample.bbox)
        return sample


class Normalize(tv.transforms.Normalize):
    def __call__(self, sample):
        sample.img = F.normalize(sample.img, self.mean, self.std, self.inplace)
        return sample


# NIH dataset loader!!!
# https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L867

# train_tx = tv.transforms.Compose([
#       tv.transforms.Resize((precrop, precrop)),
#       tv.transforms.RandomCrop((crop, crop)),
#       tv.transforms.RandomHorizontalFlip(),
#       tv.transforms.ToTensor(),
#       tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#   ])
#   val_tx = tv.transforms.Compose([
#       tv.transforms.Resize((crop, crop)),
#       tv.transforms.ToTensor(),
#       tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#   ])