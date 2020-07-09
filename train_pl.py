# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torchvision as tv
import json
import pandas as pd
from sklearn.metrics import average_precision_score

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
import bit_hyperrule
import data_utils
from data_utils import MyImagenetBoundingBoxFolder, bbox_collate, Sample, \
    MyImageFolder, MyConcatDataset, MyConcatDatasetSampler, MySubset, MyImageNetODataset
from arch.Inpainting.Baseline import RandomColorWithNoiseInpainter, LocalMeanInpainter
from arch.Inpainting.CAInpainter import CAInpainter

from argparse import ArgumentParser, Namespace
import os
import random
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.logging import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

# pull out resnet names from torchvision models
# MODEL_NAMES = sorted(
#     name for name in models.__dict__
#     if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
# )


class ImageNetLightningModel(LightningModule):
    def __init__(self, hparams):
        """
        TODO: add docstring here
        """
        super().__init__()
        self.hparams = hparams
        self.my_logger = bit_common.setup_logger(self.hparams)

        train_set, valid_sets = self.make_train_val_dataset()

        # setup the learning rate schedule for how long we want to train
        self.lr_supports = bit_hyperrule.get_schedule(len(train_set))

        # hack to make the pl train for 1 epoch = this number of steps
        train_set.my_num_samples = (self.hparams.batch * self.lr_supports[-1])

        batch_size = self.hparams.batch // self.hparams.batch_split
        self.train_loader = train_set.make_loader(
            batch_size, shuffle=True, workers=self.hparams.workers)
        self.valid_loaders = [v.make_loader(
            batch_size, shuffle=False, workers=self.hparams.workers)
            for v in valid_sets]

        if hparams.model in models.KNOWN_MODELS:
            self.model = models.KNOWN_MODELS[hparams.model](
                head_size=len(valid_sets[0].classes), zero_head=True)
            if hparams.finetune:
                self.my_logger.info("Fine-tuning from BiT")
                self.model.load_from(np.load(f"models/{hparams.model}.npz"))
        else:  # from torchvision
            raise NotImplementedError()
            # self.model = getattr(torchvision.models, hparams.model)(
            #     pretrained=hparams.finetune)

        self.mixup = 0
        if hparams.use_mixup:
            self.mixup = bit_hyperrule.get_mixup(len(train_set))

        self.inpaint_model = [self.get_inpainting_model()]

    def get_training_max_steps(self):
        return self.lr_supports[-1]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        has_bbox = isinstance(x, dict)
        if not has_bbox:
            criteron = F.cross_entropy
        else:
            criteron = self.counterfact_cri

            mask = x['imgs'].new_ones(x['imgs'].shape[0], 1, *x['imgs'].shape[2:])
            for i, (xs, ys, ws, hs) in enumerate(zip(x['xs'], x['ys'], x['ws'], x['hs'])):
                for coord_x, coord_y, w, h in zip(xs, ys, ws, hs):
                    if coord_x == -1:
                        break
                    mask[i, 0, coord_y:(coord_y + h), coord_x:(coord_x + w)] = 0.

            impute_x = self.inpaint_model[0](x['imgs'], mask)
            impute_y = (-y - 1)

            x = torch.cat([x['imgs'], impute_x], dim=0)
            # label -1 as negative of class 0, -2 as negative of class 1 etc...
            y = torch.cat([y, impute_y], dim=0)

        mixup_l = np.random.beta(self.mixup, self.mixup) if self.mixup > 0 else 1
        if self.mixup > 0.0:
            x, y_a, y_b = self.mixup_data(x, y, mixup_l)

        logits = self(x)
        if self.mixup > 0.0:
            c = self.mixup_criterion(criteron, logits, y_a, y_b, mixup_l)
        else:
            c = criteron(logits, y)

        # See clean img accuracy and cf img accuracy
        if not has_bbox:
            acc1, acc5 = self.__accuracy(logits, y, topk=(1, 5))
            cf_acc1, cf_acc5 = acc1.new_tensor(-1.), acc1.new_tensor(-1.)
        else:
            acc1, acc5 = self.__accuracy(logits[:(len(y)//2)], y[:(len(y)//2)], topk=(1, 5))

            cf_y = -(y[(len(y)//2):] + 1)
            cf_acc1, cf_acc5 = self.__accuracy(
                logits[(len(y)//2):], cf_y, topk=(1, 5))
            cf_acc1, cf_acc5 = 100. - cf_acc1, 100. - cf_acc5

        tqdm_dict = {'train_loss': c, 'train_acc1': acc1, 'train_acc5': acc5,
                     **({} if not has_bbox
                        else {'train_cf_acc1': cf_acc1, 'train_cf_acc5': cf_acc5})}
        output = OrderedDict({
            'loss': c,
            'acc1': acc1,
            'acc5': acc5,
            'cf_acc1': cf_acc1,
            'cf_acc5': cf_acc5,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
        })
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None or dataloader_idx == 0:
            images, target = batch

            output = self(images)
            loss_val = F.cross_entropy(output, target, reduction='sum')
            acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
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
            result = self.training_step(batch, batch_idx)
            output = OrderedDict({
                'val_loss': result['loss'],
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
            for metric_name in ["val_loss", "val_acc1", "val_acc5",
                                "val_cf_acc1", "val_cf_acc5"]:
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
            'gstep': self.global_step // self.hparams.batch_split, # for checkpoint filename
        }
        return result

    @classmethod
    def __accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1., momentum=0.9)
        scheduler = {
            # subtle: for lightning 0.7.x, the global step does not count the
            # accumulate_grad_batches. But after 0.8.x, the global step do count those.
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: bit_hyperrule.get_lr(
                    self.global_step // self.hparams.batch_split,
                    base_lr=self.hparams.base_lr,
                    supports=self.lr_supports)),
            'interval': 'step',
        }
        return [optimizer], [scheduler]

    @staticmethod
    def counterfact_cri(logit, y):
        if torch.all(y >= 0):
            return F.cross_entropy(logit, y, reduction='mean')

        loss1 = F.cross_entropy(logit[y >= 0], y[y >= 0], reduction='sum')

        cf_logit, cf_y = logit[y < 0], -(y[y < 0] + 1)

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
        return (loss1 + loss2) / y.shape[0]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
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

    def make_train_val_dataset(self):
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

        def sub_dataset(bbox_dataset, subset_data):
            if subset_data == 0.:
                return bbox_dataset

            num = int(subset_data)
            if subset_data <= 1.:
                num = int(len(bbox_dataset) * subset_data)

            indices = torch.randperm(len(bbox_dataset))[:num]
            bbox_dataset = MySubset(bbox_dataset, indices=indices)
            return bbox_dataset

        if self.hparams.dataset == "imagenet2012":
            train_set = MyImageFolder(pjoin(self.hparams.datadir, "train"), transform=train_tx)
            valid_sets = [MyImageFolder(pjoin(self.hparams.datadir, "val"), transform=val_tx)]
        elif self.hparams.dataset in ['objectnet', 'imageneta']:
            if self.hparams.bbox_data == 1. \
                    and self.hparams.nobbox_data == 1. \
                    and self.inpaint == 'none':
                train_set = MyImageFolder(
                    pjoin(self.hparams.datadir, f"train_{self.hparams.dataset}"),
                    transform=train_tx)
            else:
                df = pd.read_csv(pjoin(self.hparams.datadir, 'LOC_train_solution_size.csv'))
                bbox_filenames = set(df.ImageId)

                def has_bbox(path):
                    return os.path.basename(path).split('.')[0] in bbox_filenames

                ret_datasets = []
                # handle no bbox dataset
                if self.hparams.nobbox_data > 0.:
                    nobbox_d = MyImageFolder(
                        pjoin(self.hparams.datadir, f"train_{self.hparams.dataset}"),
                        is_valid_file=lambda path: ~has_bbox(path),
                        transform=train_tx)
                    nobbox_d = sub_dataset(nobbox_d, self.hparams.nobbox_data)
                    ret_datasets.append(nobbox_d)

                # handle bbox dataset
                if self.hparams.bbox_data > 0.:
                    if self.hparams.inpaint == 'none':
                        bbox_d = MyImageFolder(
                            pjoin(self.hparams.datadir, f"train_{self.hparams.dataset}"),
                            is_valid_file=has_bbox,
                            transform=train_tx)
                    else:
                        train_tx = tv.transforms.Compose([
                            data_utils.Resize((precrop, precrop)),
                            data_utils.RandomCrop((crop, crop)),
                            data_utils.RandomHorizontalFlip(),
                            data_utils.ToTensor(),
                            data_utils.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
                        bbox_d = MyImagenetBoundingBoxFolder(
                            pjoin(self.hparams.datadir, f"train_{self.hparams.dataset}"),
                            bbox_file=pjoin(self.hparams.datadir, 'LOC_train_solution.csv'),
                            is_valid_file=has_bbox,
                            transform=train_tx)
                    bbox_d = sub_dataset(bbox_d, self.hparams.bbox_data)
                    ret_datasets.insert(0, bbox_d) # training bbox data first to easily debug

                train_set = ret_datasets[0] if len(ret_datasets) == 1 \
                    else MyConcatDataset(ret_datasets)

            valid_sets = [MyImageFolder(f"../datasets/{self.hparams.dataset}/", transform=val_tx)]

            if self.hparams.val_data > 0.:
                num = int(self.hparams.val_data)
                if self.hparams.val_data <= 1.:
                    num = int(len(train_set) * self.hparams.val_data)
                shuffled_indices = torch.randperm(len(train_set))
                valid_set2 = MySubset(train_set, indices=shuffled_indices[:num])
                train_set = MySubset(train_set, indices=shuffled_indices[num:])

                valid_sets.append(valid_set2)

            if self.hparams.dataset == 'imageneta': # add an imagenet-o OOD val loader
                valid_set3 = MyImageNetODataset(
                    imageneto_dir="../datasets/imageneto/",
                    val_imgnet_dir="../datasets/imagenet/val_imageneta/",
                    transform=val_tx)
                valid_sets.append(valid_set3)

        else:
            raise ValueError(f"Sorry, we have not spent time implementing the "
                             f"{self.hparams.dataset} dataset in the PyTorch codebase. "
                             f"In principle, it should be easy to add :)")

        self.my_logger.info(f"Using a training set with {len(train_set)} images.")
        for idx, v in enumerate(valid_sets):
            self.my_logger.info(f"Using a validation set {idx} with {len(v)} images.")
        return train_set, valid_sets

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser

    def get_inpainting_model(self):
        if self.hparams.inpaint == 'none':
            inpaint_model = None
        elif self.hparams.inpaint == 'mean':
            inpaint_model = (lambda x, mask: x * mask)
        elif self.hparams.inpaint == 'random':
            inpaint_model = RandomColorWithNoiseInpainter((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        elif self.hparams.inpaint == 'local':
            inpaint_model = LocalMeanInpainter(window=15)
        elif self.hparams.inpaint == 'cagan':
            inpaint_model = CAInpainter(
                self.hparams.batch // self.hparams.batch_split,
                checkpoint_dir='./inpainting_models/release_imagenet_256/')
        else:
            raise NotImplementedError(f"Unkown inpaint {self.hparams.inpaint}")

        return inpaint_model


def get_args():
    # Big Transfer arg parser
    parser = ArgumentParser(description="Fine-tune BiT-M model.")
    parser.add_argument('--distributed_backend', type=str, default='dp',
                               choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parser.add_argument("--name", default='test',
                        help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument("--model", default='BiT-S-R50x1',
                        help="Which variant to use; BiT-M gives best results.")
    parser.add_argument("--logdir", default='./models/',
                        help="Where to log training info (small).")
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for initializing training. ')
    parser.add_argument("--examples_per_class", type=int, default=None,
                        help="For the few-shot variant, use this many examples "
                             "per class only.")
    parser.add_argument("--examples_per_class_seed", type=int, default=0,
                        help="Random seed for selecting examples.")
    parser.add_argument("--batch", type=int, default=512,
                        help="Batch size.")
    parser.add_argument("--batch_split", type=int, default=1,
                        help="Number of batches to compute gradient on before updating weights.")
    parser.add_argument("--base_lr", type=float, default=0.003,
                        help="Base learning-rate for fine-tuning. Most likely default is best.")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")

    # My own arguments
    parser.add_argument("--dataset", choices=list(bit_hyperrule.known_dataset_sizes.keys()),
                        default='imageneta',
                        help="Choose the dataset. It should be easy to add your own! "
                             "Don't forget to set --datadir if necessary.")
    parser.add_argument("--datadir", default='../datasets/imagenet',
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--bbox_data", type=float, default=1.0)
    parser.add_argument("--nobbox_data", type=float, default=0.)
    parser.add_argument("--val_data", type=float, default=1000)
    parser.add_argument("--inpaint", type=str, default='mean',
                        choices=['mean', 'random', 'local',
                                 'cagan', 'none'])
    # parser.add_argument("--bbox_max_ratio", type=float, default=1.)
    parser.add_argument("--use_mixup", type=int, default=0)  # Turn off mixup for now
    parser.add_argument("--test_run", type=int, default=1)
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--finetune", type=int, default=1)

    parser = ImageNetLightningModel.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.test_run:
        print("WATCHOUT!!! YOU ARE RUNNING IN TEST RUN MODE!!!")

        args.batch = 8
        args.batch_split = 2
        args.workers = 0
        args.eval_every = 20
    return args


def main(args: Namespace) -> None:
    model = ImageNetLightningModel(args)

    if args.seed is None:
        cudnn.benchmark = True # speed up
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    checkpoint_callback = ModelCheckpoint(
        filepath=pjoin(args.logdir, args.name, '{gstep}'),
        save_top_k=-1,
        period=-1, # a hack to save checkpts within an epoch
        verbose=True,
    )

    os.makedirs(pjoin('./lightning_logs', args.name), exist_ok=True)
    logger = TensorBoardLogger(
        save_dir='./lightning_logs/',
        name=args.name,
    )

    val_check_interval = args.eval_every
    if args.distributed_backend == 'dp':
        val_check_interval *= args.batch_split

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=1,
        # max_step=max_step,
        val_check_interval=val_check_interval,
        distributed_backend=args.distributed_backend,
        precision=16 if args.fp16 else 32,
        amp_level='O1',
        checkpoint_callback=checkpoint_callback,
        profiler=True,
        num_sanity_val_steps=1, # speed up
        accumulate_grad_batches=args.batch_split, # not much documentation. Not sure how it uses.
        logger=logger,
        benchmark=(args.seed is None),
        callbacks=[LearningRateLogger()],
        limit_val_batches=0.01 if args.test_run else 1.,
    )

    json.dump(vars(args), open(pjoin(args.logdir, args.name, 'hparams.json'), 'w'))

    # Final eval at end of training.
    if args.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())
