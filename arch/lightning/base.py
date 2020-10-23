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
from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning.callbacks import ModelCheckpoint

from .. import models, bit_common, bit_hyperrule
from ..data import bbox_utils
from ..data.imagenet_datasets import MyImagenetBoundingBoxFolder, MyImageFolder, MyConcatDataset, MySubset, \
    MyImageNetODataset, MyFactualAndCFDataset
from ..inpainting.Baseline import RandomColorWithNoiseInpainter, ShuffleInpainter, TileInpainter
from ..inpainting.AdvInpainting import AdvInpainting
from ..inpainting.VAEInpainter import VAEInpainter
from ..utils import output_csv, DotDict, generate_mask
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
        self.init_setup()

        self.inpaint = self.get_inpainting_model(self.hparams.inpaint)
        self.f_inpaint = self.get_inpainting_model(self.hparams.f_inpaint)

        # Backward compatability
        if 'cf' not in self.hparams:
            self.hparams.cf = 'logp'
        if 'result_dir' not in self.hparams:
            self.hparams.result_dir = './results/'

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, is_training=True):
        s, l = batch

        is_dict = isinstance(s, dict)
        if not is_dict:
            x, y = s, l
        else:
            orig_x_len = len(s['imgs'])
            if 'masks' in s:
                if s['masks'] is None:
                    has_bbox = s['imgs'].new_zeros(s['imgs'].shape[0]).bool()
                else:
                    mask = s['masks']
                    has_bbox = (mask == 1).any(dim=3).any(dim=2).any(dim=1)
            else:
                has_bbox = (s['xs'] != -1)
                if has_bbox.ndim == 2: # multiple bboxes
                    has_bbox = has_bbox.any(dim=1)

                mask = generate_mask(
                    s['imgs'], s['xs'], s['ys'], s['ws'], s['hs'])
            if self.hparams.inpaint == 'none' or has_bbox.sum() == 0:
                x, y = s['imgs'], l
            else:
                if 'imgs_cf' in s:
                    impute_x = s['imgs_cf'][has_bbox]
                else:
                    impute_x = self.inpaint(s['imgs'][has_bbox], 1 - mask[has_bbox])
                impute_y = (-l[has_bbox] - 1)

                x = torch.cat([s['imgs'], impute_x], dim=0)
                # label -1 as negative of class 0, -2 as negative of class 1 etc...
                y = torch.cat([l, impute_y], dim=0)

            if self.hparams.f_inpaint != 'none' and has_bbox.any():
                impute_x = self.f_inpaint(
                    s['imgs'][has_bbox], mask[has_bbox], l[has_bbox])
                x = torch.cat([x, impute_x], dim=0)
                y = torch.cat([y, l[has_bbox]], dim=0)

        if not is_dict or self.hparams.get('reg', 'none') == 'none' \
                or (is_dict and (~has_bbox).all()):
            logits = self(x)
            reg_loss = logits.new_tensor(0.)
        else:
            x_orig, y_orig = x[:orig_x_len], y[:orig_x_len]
            saliency_fn = eval(f"get_{self.hparams.get('reg_grad', 'grad_y')}")
            the_grad, logits = saliency_fn(x_orig, y_orig, self,
                                           is_training=is_training)
            if torch.all(the_grad == 0.):
                reg_loss = logits.new_tensor(0.)
            elif self.hparams.reg == 'gs':
                assert is_dict and self.hparams.inpaint != 'none'
                if not has_bbox.any():
                    reg_loss = logits.new_tensor(0.)
                else:
                    x_orig, x_cf = x[:orig_x_len], x[orig_x_len:(orig_x_len + has_bbox.sum())]
                    dist = x_orig[has_bbox] - x_cf
                    cos_sim = self.my_cosine_similarity(the_grad, dist)

                    reg_loss = (1. - cos_sim).mean().mul_(self.hparams.reg_coeff)
            elif self.hparams.reg == 'bbox_o':
                reg_loss = ((the_grad[has_bbox] * (1 - mask[has_bbox])) ** 2).mean()\
                    .mul_(self.hparams.reg_coeff)
            elif self.hparams.reg == 'bbox_f1':
                norm = (the_grad[has_bbox] ** 2).sum(dim=1, keepdim=True)
                norm = (norm - norm.min()) / norm.max()
                gnd_truth = mask[has_bbox]

                f1 = self.diff_f1_score(norm, gnd_truth)
                # (1 - F1) as the loss
                reg_loss = (1. - f1).mul_(self.hparams.reg_coeff)
            else:
                raise NotImplementedError(self.hparams.reg)

            # Doing annealing for reg loss
            if self.hparams.reg_anneal > 0.:
                anneal = self.global_step / (self.hparams.max_epochs * len(self.train_loader)
                                             * self.hparams.reg_anneal)
                reg_loss *= anneal

            if len(x) > orig_x_len: # Other f or cf images
                cf_logits = self(x[orig_x_len:])
                logits = torch.cat([logits, cf_logits], dim=0)

        c, c_cf = self.counterfact_cri(logits, y)
        c_cf *= self.hparams.cf_coeff

        # See clean img accuracy and cf img accuracy
        if not is_dict or (~has_bbox).all() or self.hparams.inpaint == 'none':
            acc1, = self.accuracy(logits, y, topk=(1,))
            cf_acc1 = acc1.new_tensor(-1.)
        else:
            acc1, = self.accuracy(
                logits[:orig_x_len], y[:orig_x_len], topk=(1,))

            cf_y = -(y[orig_x_len:] + 1)
            cf_acc1, = self.accuracy(
                logits[orig_x_len:], cf_y, topk=(1,))
            cf_acc1 = 100. - cf_acc1

        # Check NaN
        for name, metric in [
            ('train_loss', c),
            ('cf_loss', c_cf),
            ('reg_loss', reg_loss),
        ]:
            if torch.isnan(metric).all():
                raise RuntimeError(f'metric {name} is Nan')

        tqdm_dict = {'train_loss': c,
                     'cf_loss': c_cf,
                     'reg_loss': reg_loss,
                     'train_acc1': acc1,
                     'train_cf_acc1': cf_acc1}
        output = OrderedDict({
            'loss': c + c_cf + reg_loss,
            'train_loss': c,
            'cf_loss': c_cf,
            'reg_loss': reg_loss,
            'acc1': acc1,
            'cf_acc1': cf_acc1,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
        })
        return output

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
            loss = -F.log_softmax(logit, dim=1).mean(dim=1)
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
            alpha = self.hparams.alpha
            if alpha == -1:
                alpha = self.hparams.eps * 1.25
            inpaint_model = AdvInpainting(
                self.model, eps=self.hparams.eps,
                alpha=alpha,
                attack=inpaint)
        else:
            raise NotImplementedError(f"Unkown inpaint {inpaint}")

        return inpaint_model

    def test_dataloader(self):
        '''
        This is for OOD detections. The first loader is the normal
        test set. And the rest of the loaders are from other datasets
        like Gaussian, Uniform, CCT and Xray.

        Gaussian, Uniform: generate the same number as test sets (45k)
        CCT: whole dataset like 45k?
        Xray: 30k?
        '''
        test_sets = self._make_test_datasets()
        if test_sets is None:
            return None

        for idx, (n, v) in enumerate(zip(self.test_sets_names, test_sets)):
            self.my_logger.info(f"Using a test set {idx} {n} with {len(v)} images.")

        test_loaders = [v.make_loader(
            self.hparams.val_batch, shuffle=False, workers=self.hparams.workers)
            for v in test_sets]
        return test_loaders

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        logits = self(x)
        prob = F.softmax(logits, dim=1)
        anomaly_score = 1. - (prob.max(dim=1).values)

        output = {
            'anomaly_score': anomaly_score,
        }
        return output

    def test_epoch_end(self, outputs):
        tqdm_dict = {}

        def cal_metrics(the_test, the_orig, prefix='gn'):
            the_as = torch.cat([o['anomaly_score'] for o in the_test])
            orig_as = torch.cat([o['anomaly_score'] for o in the_orig])

            # 95% TPR: k is the kth-smallest element
            # I want 95% of examples to below this number
            thresh = torch.kthvalue(
                orig_as,
                k=int(np.floor(0.95 * len(orig_as)))).values
            fpr = (the_as <= thresh).float().mean().item()
            tqdm_dict[f'{prefix}_ood_fpr'] = fpr

            cat_as = torch.cat([the_as, orig_as], dim=0)
            ys = torch.cat([torch.ones(len(the_as)), torch.zeros(len(orig_as))], dim=0)
            tqdm_dict[f'{prefix}_ood_auc'] = roc_auc_score(
                ys.cpu().numpy(), cat_as.cpu().numpy())
            tqdm_dict[f'{prefix}_ood_aupr'] = average_precision_score(
                ys.cpu().numpy(), cat_as.cpu().numpy())

        for name, output in zip(self.test_sets_names[1:], outputs[1:]):
            cal_metrics(output, outputs[0], name)

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict,
        }
        return result

    def _make_test_datasets(self):
        raise None

    @classmethod
    def add_model_specific_args(cls, parser):  # pragma: no-cover
        raise NotImplementedError()

    def pl_trainer_args(self):
        raise NotImplementedError()

    @classmethod
    def is_finished_run(cls, model_dir):
        raise NotImplementedError()


class EpochBaseLightningModel(BaseLightningModel):
    @classmethod
    def is_finished_run(cls, model_dir):
        last_ckpt = pjoin(model_dir, 'last.ckpt')
        if pexists(last_ckpt):
            tmp = torch.load(last_ckpt,
                             map_location=torch.device('cpu'))
            last_epoch = tmp['epoch']
            hparams = tmp['hyper_parameters']
            if last_epoch >= hparams.max_epochs:
                print('Already finish fitting! Max %d Last %d'
                      % (hparams.max_epochs, last_epoch))
                return True

        return False
