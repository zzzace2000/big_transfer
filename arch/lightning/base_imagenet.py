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
from ..inpainting.Baseline import RandomColorWithNoiseInpainter, ShuffleInpainter
from ..inpainting.VAEInpainter import VAEInpainter
from ..utils import output_csv, DotDict
from ..saliency_utils import get_grad_y, get_grad_sum, \
    get_grad_logp_sum, get_grad_logp_y, get_deeplift


KNOWN_NUM_CLASSES = {
    'objectnet': 113,
    'imageneta': 200,
}


class ImageNetLightningModel(LightningModule):
    def __init__(self, hparams):
        """
        Training imagenet models by fintuning from Big-Transfer models
        """
        super().__init__()
        if isinstance(hparams, dict): # Fix the bug in pl in reloading
            hparams = DotDict(hparams)
        if 'result_dir' not in hparams:
            hparams.result_dir = './results/'
        self.hparams = hparams
        self.my_logger = bit_common.setup_logger(self.hparams)
        self.train_loader = None
        self.valid_loaders = None

        self.inpaint_model = self.get_inpainting_model()
        self.init_setup()

    def init_setup(self):
        # setup the learning rate schedule for how long we want to train
        self.lr_supports = [500, 3000, 6000, 9000, 10_000]

        if self.hparams.model in models.KNOWN_MODELS:
            self.model = models.KNOWN_MODELS[self.hparams.model](
                head_size=KNOWN_NUM_CLASSES[self.hparams.dataset],
                zero_head=True)
            if self.hparams.finetune:
                self.my_logger.info("Fine-tuning from BiT")
                self.model.load_from(np.load(f"models/{self.hparams.model}.npz"))
        else:  # from torchvision
            raise NotImplementedError()

        self.mixup = 0
        # if hparams.use_mixup:
        #     self.mixup = bit_hyperrule.get_mixup(len(train_set))

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        if self.global_step >= self.lr_supports[-1]:
            self.my_logger.info('Already finish training! Exit!')
            exit()

    def training_step(self, batch, batch_idx):
        return self._training_step(batch, batch_idx)

    def _training_step(self, batch, batch_idx, is_training=True):
        x, y = batch

        is_dict = isinstance(x, dict)
        if is_dict:
            if 'imgs_cf' in x:
                bbox = {k: x[k] for k in ['xs', 'ys', 'ws', 'hs']}
                x = torch.cat([x['imgs'], x['imgs_cf']], dim=0)
                y = torch.cat([y, (-y - 1)], dim=0)
                mask = lambda imgs: self._generate_mask(imgs, **bbox) # lazy loading
            else:
                mask = self._generate_mask(
                    x['imgs'], x['xs'], x['ys'], x['ws'], x['hs'])
                if self.hparams.inpaint == 'none':
                    x = x['imgs']
                else:
                    impute_x = self.inpaint_model(x['imgs'], mask)
                    impute_y = (-y - 1)

                    x = torch.cat([x['imgs'], impute_x], dim=0)
                    # label -1 as negative of class 0, -2 as negative of class 1 etc...
                    y = torch.cat([y, impute_y], dim=0)

        if self.hparams.get('reg', 'none') == 'none':
            logits = self(x)
            reg_loss = logits.new_tensor(0.)
        else:
            if is_dict and self.hparams.inpaint != 'none':
                half = len(x) // 2
                x_orig, x_cf, y_orig = x[:half], x[half:], y[:half]
            else:
                x_orig, y_orig = x, y

            saliency_fn = eval(f"get_{self.hparams.get('reg_grad', 'grad_y')}")
            the_grad, logits = saliency_fn(x_orig, y_orig, self.model,
                                           is_training=is_training)
            if torch.all(the_grad == 0.):
                reg_loss = logits.new_tensor(0.)
            elif self.hparams.reg == 'gs':
                assert is_dict and self.hparams.inpaint != 'none'
                dist = x_orig.detach() - x_cf
                cos_sim = self.my_cosine_similarity(the_grad, dist)

                reg_loss = (1. - cos_sim).mean().mul_(self.hparams.reg_coeff)
            elif self.hparams.reg == 'bbox_o': # 'bbox_o'
                # a simple loss that the saliency outside of the box is bad
                if callable(mask):
                    mask = mask(x_orig)

                reg_loss = ((the_grad * mask) ** 2).mean()\
                    .mul_(self.hparams.reg_coeff)
            elif self.hparams.reg == 'bbox_f1':
                if callable(mask):
                    mask = mask(x_orig)
                norm = (the_grad ** 2).sum(dim=1, keepdim=True)
                norm = (norm - norm.min()) / norm.max()
                gnd_truth = (1. - mask)

                f1 = self.diff_f1_score(norm, gnd_truth)
                # (1 - F1) as the loss
                reg_loss = (1. - f1).mul_(self.hparams.reg_coeff)
            else:
                raise NotImplementedError(self.hparams.reg)

            # Doing annealing
            if self.hparams.reg_anneal > 0. \
                    and self.global_step < \
                    self.hparams.reg_anneal * self.lr_supports[-1]:
                reg_loss *= (self.global_step / self.lr_supports[-1]
                             / self.hparams.reg_anneal)

            if is_dict and self.hparams.inpaint != 'none':
                cf_logits = self(x_cf)
                logits = torch.cat([logits, cf_logits], dim=0)

        c = self.counterfact_cri(logits, y)

        # See clean img accuracy and cf img accuracy
        if not is_dict:
            acc1, acc5 = self.accuracy(logits, y, topk=(1, 5))
            cf_acc1, cf_acc5 = acc1.new_tensor(-1.), acc1.new_tensor(-1.)
        else:
            acc1, acc5 = self.accuracy(
                logits[:(len(y)//2)], y[:(len(y)//2)], topk=(1, 5))

            cf_y = -(y[(len(y)//2):] + 1)
            cf_acc1, cf_acc5 = self.accuracy(
                logits[(len(y)//2):], cf_y, topk=(1, 5))
            cf_acc1, cf_acc5 = 100. - cf_acc1, 100. - cf_acc5

        tqdm_dict = {'train_loss': c, 'reg_loss': reg_loss, 'train_acc1': acc1, 'train_acc5': acc5,
                     **({} if not is_dict
                        else {'train_cf_acc1': cf_acc1, 'train_cf_acc5': cf_acc5})}
        output = OrderedDict({
            'loss': c + reg_loss,
            'train_loss': c,
            'reg_loss': reg_loss,
            'acc1': acc1,
            'acc5': acc5,
            'cf_acc1': cf_acc1,
            'cf_acc5': cf_acc5,
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

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None or dataloader_idx == 0:
            images, target = batch

            output = self(images)
            loss_val = F.cross_entropy(output, target, reduction='sum')
            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
            batch_size = images.new_tensor(images.shape[0])

            output = OrderedDict({
                'val_loss': loss_val,
                'val_acc1': acc1 * batch_size,
                'val_acc5': acc5 * batch_size,
                'batch_size': batch_size,
            })
            return output

        # handle the second loader for part of the train loader
        # since counting cf size is annoying, just take avg
        if dataloader_idx == 1:
            result = self._training_step(batch, batch_idx,
                                         is_training=False)
            output = OrderedDict({
                'val_loss': result['train_loss'],
                'val_reg_loss': result['reg_loss'],
                'val_acc1': result['acc1'],
                'val_acc5': result['acc5'],
                'val_cf_acc1': result['cf_acc1'],
                'val_cf_acc5': result['cf_acc5'],
            })
            return output

        # for imagenet-o
        if dataloader_idx == 2:
            x, y = batch
            logits = self(x)
            anomaly_score = -(logits.max(dim=1).values)

            output = OrderedDict({
                'as': anomaly_score,
                'y': y,
            })
            return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}

        # 1st val loader
        the_outputs = outputs if isinstance(outputs[0], dict) else outputs[0]
        all_size = torch.stack([o['batch_size'] for o in the_outputs]).sum()
        for metric_name in ["val_loss", "val_acc1", "val_acc5"]:
            metrics = [o[metric_name] for o in the_outputs]
            tqdm_dict[metric_name] = torch.sum(torch.stack(metrics)) / all_size

        # 2nd val loader
        if isinstance(outputs[0], list) and len(outputs) > 1:
            the_outputs = outputs[1]
            for metric_name in ["val_loss", "val_reg_loss", "val_acc1",
                                "val_acc5", "val_cf_acc1", "val_cf_acc5"]:
                metrics = [o[metric_name] for o in the_outputs]
                metrics = torch.stack(metrics)
                metrics = metrics[metrics >= 0.] # filter out neg value
                tqdm_dict[metric_name + '_train'] = metrics.mean()

        # 3rd val loader
        if isinstance(outputs[0], list) and len(outputs) > 2:
            the_outputs = outputs[2]
            anomaly_scores = torch.cat([o['as'] for o in the_outputs])
            ys = torch.cat([o['y'] for o in the_outputs])

            tqdm_dict['imgneto_aupr'] = average_precision_score(
                ys.cpu().numpy(), anomaly_scores.cpu().numpy())

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"],
            'gstep': self.global_step, # forresult_dir checkpoint filename
        }

        # Record it in the last training step
        if self.global_step >= self.lr_supports[-1] - 1:
            csv_dict = OrderedDict()
            csv_dict['name'] = self.hparams.name
            csv_dict.update(tqdm_dict)
            csv_dict.update(
                vars(self.hparams) if isinstance(self.hparams, Namespace)
                else self.hparams)
            output_csv(pjoin(self.hparams.result_dir, 'results.csv'), csv_dict)

        return result

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
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1., momentum=0.9)
        scheduler = {
            # subtle: for lightning 0.7.x, the global step does not count the
            # accumulate_grad_batches. But after 0.8.x, the global step do count those.
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: bit_hyperrule.get_lr(
                    self.global_step,
                    base_lr=self.hparams.base_lr,
                    supports=self.lr_supports)),
            'interval': 'step',
        }
        return [optimizer], [scheduler]

    @classmethod
    def counterfact_cri(cls, logit, y):
        if torch.all(y >= 0):
            return F.cross_entropy(logit, y, reduction='mean')
        if torch.all(y < 0):
            return cls.counterfactual_ce_loss(logit, y) / y.shape[0]

        loss1 = F.cross_entropy(logit[y >= 0], y[y >= 0], reduction='sum')
        loss2 = cls.counterfactual_ce_loss(logit[y < 0], y[y < 0])

        return (loss1 + loss2) / y.shape[0]

    @classmethod
    def counterfactual_ce_loss(cls, logit, y):
        assert (y < 0).all(), str(y)
        cf_logit, cf_y = logit, -(y + 1)

        if cf_logit.shape[1] == 2: # 2-cls
            return F.cross_entropy(logit, 1 - cf_y, reduction='mean')

        # Implement my own logsumexp trick
        m, _ = torch.max(cf_logit, dim=1, keepdim=True)
        exp_logit = torch.exp(cf_logit - m)
        sum_exp_logit = torch.sum(exp_logit, dim=1)

        eps = 1e-20
        num = (sum_exp_logit - exp_logit[torch.arange(exp_logit.shape[0]), cf_y])
        num = torch.log(num + eps)
        denon = torch.log(sum_exp_logit + eps)

        # Negative log probability
        loss2 = -(num - denon).sum()
        return loss2

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
        precrop, crop = bit_hyperrule.get_resolution_from_dataset(self.hparams.dataset)
        if self.hparams.test_run:  # save memory
            precrop, crop = 64, 56

        train_tx = tv.transforms.Compose([
            tv.transforms.Resize((precrop, precrop)),
            tv.transforms.RandomCrop((crop, crop)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        val_tx = tv.transforms.Compose([
            tv.transforms.Resize((crop, crop)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        valid_sets = [MyImageFolder(f"../datasets/{self.hparams.dataset}/", transform=val_tx)]

        df = pd.read_csv(pjoin(self.hparams.datadir, 'LOC_train_solution_size.csv'))
        def has_bbox(bbox_max_ratio=1., inverse=False):
            bbox_filenames = set(df[df.bbox_ratio <= bbox_max_ratio].ImageId)
            def is_valid_file(path):
                ans = os.path.basename(path).split('.')[0] in bbox_filenames
                if inverse:
                    ans = (not ans)
                return ans
            return is_valid_file

        train_sets = []
        # handle bbox dataset
        if self.hparams.bbox_data > 0.:
            train_bbox_tx = tv.transforms.Compose([
                bbox_utils.Resize((precrop, precrop)),
                bbox_utils.RandomCrop((crop, crop)),
                bbox_utils.RandomHorizontalFlip(),
                bbox_utils.ToTensor(),
                bbox_utils.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            if self.hparams.inpaint == 'none' and self.hparams.reg == 'none':
                bbox_d = MyImageFolder(
                    pjoin(self.hparams.datadir, f"train_{self.hparams.dataset}"),
                    is_valid_file=has_bbox(self.hparams.bbox_max_ratio),
                    transform=train_tx)
            else:
                bbox_d = MyImagenetBoundingBoxFolder(
                    pjoin(self.hparams.datadir, f"train_{self.hparams.dataset}"),
                    bbox_file=pjoin(self.hparams.datadir, 'LOC_train_solution.csv'),
                    is_valid_file=has_bbox(self.hparams.bbox_max_ratio),
                    transform=train_bbox_tx)
                if self.hparams.inpaint == 'cagan':
                    cagan_d = MyImageFolder(
                        pjoin(self.hparams.datadir, f"train_cagan_{self.hparams.dataset}"),
                        is_valid_file=has_bbox(self.hparams.bbox_max_ratio),
                        transform=train_tx)
                    bbox_d = MyFactualAndCFDataset(bbox_d, cagan_d)

            bbox_d, _ = self.sub_dataset(bbox_d, self.hparams.bbox_data)
            train_sets.append(bbox_d)

        # handle no bbox dataset
        bbox_len = 0. if len(train_sets) == 0 else sum([len(d) for d in train_sets])
        self.my_logger.info(f"bbox length {bbox_len}")
        if self.hparams.nobbox_data > 0.:
            nobbox_d = MyImageFolder(
                pjoin(self.hparams.datadir, f"train_{self.hparams.dataset}"),
                is_valid_file=has_bbox(inverse=True),
                transform=train_tx)

            nobbox_data = self.hparams.nobbox_data
            if nobbox_data <= 1.: # the ratio relative to the bbox data
                assert self.hparams.bbox_data > 0.
                nobbox_data = int(bbox_len * nobbox_data)
            nobbox_d, _ = self.sub_dataset(nobbox_d, nobbox_data)
            self.my_logger.info(f"nobbox length {len(nobbox_d)}")
            train_sets.append(nobbox_d)

        train_set = train_sets[0] if len(train_sets) == 1 \
            else MyConcatDataset(train_sets)

        ## Add a second validation set
        assert self.hparams.val_data > 0.
        if not isinstance(train_set, MyConcatDataset) or not train_set.use_my_batch_sampler:
            valid_set2, train_set = self.sub_dataset(train_set, self.hparams.val_data)
        else:
            new_train_set, new_valid_set2 = [], []
            for d in train_set.datasets:
                v, t = self.sub_dataset(d, self.hparams.val_data)
                new_train_set.append(t)
                new_valid_set2.append(v)
            train_set = MyConcatDataset(new_train_set)
            valid_set2 = MyConcatDataset(new_valid_set2)

        valid_sets.append(valid_set2)

        if self.hparams.dataset == 'imageneta': # add an imagenet-o OOD val loader
            valid_set3 = MyImageNetODataset(
                imageneto_dir="../datasets/imageneto/",
                val_imgnet_dir="../datasets/imagenet/val_imageneto/",
                transform=val_tx)
            valid_sets.append(valid_set3)

        # Hack to make the pl train for 1 epoch = this number of steps.
        # This hack does not work since inpainting would only need 50%
        # samples to get. So this is just an upper bound.
        train_set.my_num_samples = \
            (self.hparams.batch * (self.lr_supports[-1]))
        return train_set, valid_sets

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

    def get_inpainting_model(self):
        if self.hparams.inpaint in ['none', 'cagan']:
            inpaint_model = None
        elif self.hparams.inpaint == 'mean':
            inpaint_model = (lambda x, mask: x * mask)
        elif self.hparams.inpaint == 'random':
            inpaint_model = RandomColorWithNoiseInpainter((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif self.hparams.inpaint == 'shuffle':
            inpaint_model = ShuffleInpainter()
        elif self.hparams.inpaint == 'vae':
            inpaint_model = VAEInpainter(in_mean=(0.5, 0.5, 0.5), in_std=(0.5, 0.5, 0.5))
            gen_model_path = './inpainting_models/0928-VAE-Var-hole_lr_0.0002_epochs_7'
            inpaint_model.load_state_dict(
                torch.load(gen_model_path, map_location=lambda storage, loc: storage)['state_dict'],
                strict=False)
        else:
            raise NotImplementedError(f"Unkown inpaint {self.hparams.inpaint}")

        return inpaint_model

    def on_load_checkpoint(self, checkpoint):
        ''' Fix the bug after loading the global step is not set '''
        self.global_step = checkpoint['global_step']

    @classmethod
    def add_model_specific_args(cls, parser):  # pragma: no-cover
        parser.add_argument("--bbox_data", type=float, default=1.0)
        parser.add_argument("--bbox_max_ratio", type=float, default=1.)
        parser.add_argument("--nobbox_data", type=float, default=0.)

        parser.add_argument("--batch", type=int, default=128,
                            help="Batch size.")
        parser.add_argument("--batch_split", type=int, default=1,
                            help="Number of batches to compute gradient on before updating weights.")
        parser.add_argument("--base_lr", type=float, default=0.003,
                            help="Base learning-rate for fine-tuning. Most likely default is best.")
        parser.add_argument("--eval_every", type=int, default=1000,
                            help="Run prediction on validation set every so many steps."
                                 "Will always run one evaluation at the end of training.")
        parser.add_argument("--val_data", type=float, default=2000)
        parser.add_argument("--test_data", type=float, default=2000)
        parser.add_argument("--pl_model", type=str, default=cls.__name__)
        parser.add_argument("--reg_anneal", type=float, default=0.)
        return parser

    # @classmethod
    # def init_model(cls, args):
    #     fs = os.listdir(pjoin(args.logdir, args.name))
    #     gsteps = [int(f.split('.')[0][6:])
    #               for f in fs if f.startswith("gstep")]
    #     if len(gsteps) == 0:
    #         model = ImageNetLightningModel(args)
    #     else:
    #         # find the largest gstep
    #         max_gstep = int(np.max(gsteps))
    #         model = ImageNetLightningModel.load_from_checkpoint(
    #             pjoin(args.logdir, args.name, 'gstep=%d.ckpt' % max_gstep)
    #         )
    #         model.my_logger.info('Loading models from gstep %d' % max_gstep)
    #         if max_gstep == model.lr_supports[-1] - 1:
    #             model.my_logger.info('Already finish this training!')
    #             exit()
    #     return model

    def pl_trainer_args(self):
        checkpoint_callback = ModelCheckpoint(
            filepath=pjoin(self.hparams.logdir, self.hparams.name, '{gstep}'),
            save_top_k=-1,
            save_last=True,
            period=-1,  # -1 then it saves checkpts within an epoch
            verbose=True,
        )

        val_check_interval = self.hparams.eval_every
        if self.hparams.distributed_backend == 'dp':
            val_check_interval *= self.hparams.batch_split

        args = dict()
        args['max_steps'] = (self.lr_supports[-1] - self.global_step - 1)
        args['max_epochs'] = 1
        args['val_check_interval'] = val_check_interval
        args['checkpoint_callback'] = checkpoint_callback

        last_ckpt = pjoin(self.hparams.logdir, self.hparams.name, 'last.ckpt')
        if pexists(last_ckpt):
            args['resume_from_checkpoint'] = last_ckpt
        return args
