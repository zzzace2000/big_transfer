import os
from collections import OrderedDict
from os.path import join as pjoin  # pylint: disable=g-importing-member
from os.path import exists as pexists

from argparse import Namespace
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.data.distributed
import torchvision as tv
from pytorch_lightning.core import LightningModule
from sklearn.metrics import average_precision_score
from pytorch_lightning.callbacks import ModelCheckpoint

from .. import models, bit_common, bit_hyperrule
from ..data import bbox_utils
from ..data.imagenet_datasets import MyImagenetBoundingBoxFolder, MyImageFolder, MyConcatDataset, MySubset, \
    MyImageNetODataset, MyFactualAndCFDataset
from ..inpainting.Baseline import RandomColorWithNoiseInpainter, ShuffleInpainter, TileInpainter
from ..inpainting.AdvInpainting import AdvInpainting
from ..inpainting.VAEInpainter import VAEInpainter
from ..utils import output_csv, DotDict
from ..saliency_utils import get_grad_y, get_grad_sum, \
    get_grad_logp_sum, get_grad_logp_y, get_deeplift, get_grad_cam


class BaseLightningModel(LightningModule):
    def __init__(self, hparams):
        """
        Training imagenet models by fintuning from Big-Transfer models
        """
        super().__init__()
        if isinstance(hparams, dict): # Fix the bug in pl in reloading
            hparams = DotDict(hparams)
        self.hparams = hparams
        self.my_logger = bit_common.setup_logger(self.hparams)
        self.train_loader = None
        self.valid_loaders = None

        self.inpaint = self.get_inpainting_model(self.hparams.inpaint)
        self.f_inpaint = self.get_f_inpainting_model(self.hparams.f_inpaint)
        self.init_setup()

        # Backward compatability
        if 'cf' not in self.hparams:
            self.hparams.cf = 'logp'
        if 'result_dir' not in self.hparams:
            self.hparams.result_dir = './results/'

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def my_cosine_similarity(t1, t2, eps=1e-8):
        if torch.all(t1 == 0.) or torch.all(t2 == 0.):
            return t1.new_tensor(0.)
        other_dim = list(range(1, t1.ndim))
        iprod = (t1 * t2).sum(dim=other_dim)
        t1_norm = (t1 * t1).sum(dim=other_dim).sqrt()
        t2_norm = (t2 * t2).sum(dim=other_dim).sqrt()
        cos_sim = iprod / (t1_norm * t2_norm + eps)
        return cos_sim

    @staticmethod
    def diff_f1_score(pred, gnd_truth):
        TP = pred.mul(gnd_truth).sum(dim=list(range(1, pred.ndim)))
        FP = pred.mul(1. - gnd_truth).sum(dim=list(range(1, pred.ndim)))
        FN = (1. - pred).mul(gnd_truth).sum(dim=list(range(1, pred.ndim)))

        # (1 - F1) as the loss
        return (2 * TP / (2 * TP + FP + FN)).mean()

    @classmethod
    def accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @classmethod
    def _generate_mask(cls, imgs, xs, ys, ws, hs):
        mask = imgs.new_ones(imgs.shape[0], 1, *imgs.shape[2:])
        for i, (xs, ys, ws, hs) in enumerate(zip(xs, ys, ws, hs)):
            if xs.ndim == 0:
                if xs > 0:
                    mask[i, 0, ys:(ys + hs), xs:(xs + ws)] = 0.
                continue

            for coord_x, coord_y, w, h in zip(xs, ys, ws, hs):
                if coord_x == -1:
                    break
                mask[i, 0, coord_y:(coord_y + h), coord_x:(coord_x + w)] = 0.
        return mask

    def configure_optimizers(self):
        raise NotImplementedError()

    def counterfact_cri(self, logit, y):
        '''
        :return: (avg normal loss, avg ce_loss)
        '''
        zero_tensor = logit.new_tensor(0.)
        if torch.all(y >= 0):
            return F.cross_entropy(logit, y, reduction='mean'), zero_tensor
        if torch.all(y < 0):
            return zero_tensor, self.counterfactual_ce_loss(logit, y, reduction='mean')

        loss1 = F.cross_entropy(logit[y >= 0], y[y >= 0], reduction='mean')
        loss2 = self.counterfactual_ce_loss(logit[y < 0], y[y < 0], reduction='mean')

        return loss1, loss2

    def counterfactual_ce_loss(self, logit, y, reduction='none'):
        assert (y < 0).all(), str(y)

        cf_y = -(y + 1)
        if self.hparams.cf == 'uni': # uniform prob
            loss = F.log_softmax(logit, dim=1).mean(dim=1)
        elif self.hparams.cf == 'uni_e': # uniform prob except the cls
            logp = F.log_softmax(logit, dim=1)
            weights = torch.ones_like(logp).mul_(1. / (
                    logp.shape[1] - 1))
            weights[torch.arange(len(cf_y)), cf_y] = 0.
            loss = -(weights * logp).sum(dim=1)
        elif self.hparams.cf == 'logp':
            if logit.shape[1] == 2: # 2-cls
                return F.cross_entropy(logit, 1 - cf_y, reduction=reduction)

            # Implement my own logsumexp trick
            m, _ = torch.max(logit, dim=1, keepdim=True)
            exp_logit = torch.exp(logit - m)
            sum_exp_logit = torch.sum(exp_logit, dim=1)

            eps = 1e-20
            num = (sum_exp_logit - exp_logit[torch.arange(exp_logit.shape[0]), cf_y])
            num = torch.log(num + eps)
            denon = torch.log(sum_exp_logit + eps)

            # Negative log probability
            loss = -(num - denon)
        else:
            raise NotImplementedError(str(self.hparams.cf))

        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()

    def train_dataloader(self):
        if self.train_loader is None:
            self._setup_loaders()
        return self.train_loader

    def val_dataloader(self):
        if self.valid_loaders is None:
            self._setup_loaders()
        return self.valid_loaders

    @staticmethod
    def mixup_data(x, y, l):
        """Returns mixed inputs, pairs of targets, and lambda"""
        indices = torch.randperm(x.shape[0]).to(x.device)

        mixed_x = l * x + (1 - l) * x[indices]
        y_a, y_b = y, y[indices]
        return mixed_x, y_a, y_b

    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, l):
        return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

    def _setup_loaders(self):
        train_set, valid_sets = self._make_train_val_dataset()
        self.my_logger.info(f"Using a training set with {len(train_set)} images.")
        for idx, v in enumerate(valid_sets):
            self.my_logger.info(f"Using a validation set {idx} with {len(v)} images.")

        train_bs = self.hparams.batch // self.hparams.batch_split
        val_bs = train_bs
        if 'val_batch' in self.hparams:
            val_bs = self.hparams.val_batch // self.hparams.batch_split
        self.train_loader = train_set.make_loader(
            train_bs, shuffle=True, workers=self.hparams.workers)
        self.valid_loaders = [v.make_loader(
            val_bs, shuffle=False, workers=self.hparams.workers)
            for v in valid_sets]

    def _make_train_val_dataset(self):
        raise NotImplementedError()

    @staticmethod
    def sub_dataset(bbox_dataset, subset_data, sec_dataset=None):
        if sec_dataset is not None:
            assert len(sec_dataset) == len(bbox_dataset)

        if subset_data == 0.:
            if sec_dataset is None:
                return None, bbox_dataset
            return None, bbox_dataset, None, sec_dataset
        if subset_data == 1. or subset_data >= len(bbox_dataset):
            if sec_dataset is None:
                return bbox_dataset, None
            return bbox_dataset, None, sec_dataset, None

        num = int(subset_data)
        if subset_data < 1.:
            num = int(len(bbox_dataset) * subset_data)

        indices = torch.randperm(len(bbox_dataset))
        first_dataset = MySubset(bbox_dataset, indices=indices[:num])
        rest_dataset = MySubset(bbox_dataset, indices=indices[num:])
        if sec_dataset is None:
            return first_dataset, rest_dataset

        fs = MySubset(sec_dataset, indices=indices[:num])
        rs = MySubset(sec_dataset, indices=indices[num:])
        return first_dataset, rest_dataset, fs, rs

    def get_inpainting_model(self, inpaint):
        if inpaint in ['none']:
            inpaint_model = None
        elif inpaint == 'mean':
            inpaint_model = (lambda x, mask: x * mask)
        elif inpaint == 'random':
            inpaint_model = RandomColorWithNoiseInpainter((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif inpaint == 'shuffle':
            inpaint_model = ShuffleInpainter()
        elif inpaint == 'tile':
            inpaint_model = TileInpainter()
        elif inpaint in ['pgd', 'fgsm']:
            inpaint_model = AdvInpainting(
                self.model, eps=self.hparams.eps,
                alpha=self.hparams.alpha,
                attack=inpaint)
        else:
            raise NotImplementedError(f"Unkown inpaint {inpaint}")

        return inpaint_model

    @classmethod
    def add_model_specific_args(cls, parser):  # pragma: no-cover
        raise NotImplementedError()

    def pl_trainer_args(self):
        raise NotImplementedError()
