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

import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
import bit_hyperrule
import models_utils
import data_utils
from data_utils import MyImagenetBoundingBoxFolder, bbox_collate, Sample, MyImageFolder
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
import torchvision
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
        # self.chrono = lb.Chrono()

        n_train, n_classes, train_loader, val_loader = self.mktrainval()
        # Use Bit-rule to select lr_schedules
        # Then set the number to make the loader 1 epoch = max steps

        self.train_loader = train_loader
        self.val_loader = val_loader

        if hparams.model in models.KNOWN_MODELS:
            self.model = models.KNOWN_MODELS[hparams.model](
                head_size=n_classes, zero_head=True)
            if hparams.finetune:
                self.my_logger.info("Fine-tuning from BiT")
                self.model.load_from(np.load(f"models/{hparams.model}.npz"))
        else:  # from torchvision
            self.model = getattr(torchvision.models, hparams.model)(
                pretrained=hparams.finetune)

        self.mixup = 0
        if hparams.use_mixup:
            self.mixup = bit_hyperrule.get_mixup(n_train)

        self.inpaint_model = [self.get_inpainting_model()]

    def get_training_max_steps(self):
        return self.lr_supports[-1]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if not isinstance(x, Sample):
            criteron = F.cross_entropy
        else:
            criteron = self.counterfact_cri

            bboxes = x.bbox
            x = x.img

            mask = x.new_ones(x.shape[0], 1, *x.shape[2:])
            for i, bbox in enumerate(bboxes):
                for coord_x, coord_y, w, h in zip(bbox.xs, bbox.ys, bbox.ws, bbox.hs):
                    mask[i, 0, coord_y:(coord_y + h), coord_x:(coord_x + w)] = 0.

            impute_x = self.inpaint_model[0](x, mask)
            impute_y = (-y - 1)

            x = torch.cat([x, impute_x], dim=0)
            # label -1 as negative of class 0, -2 as negative of class 1 etc...
            y = torch.cat([y, impute_y], dim=0)

        # if self.use_dp:
        #     print(x.device)
        #     device = torch.device("cuda:0")
        #     x = x.to(device, non_blocking=True)
        #     y = y.to(device, non_blocking=True)

        mixup_l = np.random.beta(self.mixup, self.mixup) if self.mixup > 0 else 1
        if self.mixup > 0.0:
            x, y_a, y_b = self.mixup_data(x, y, mixup_l)

        logits = self(x)
        if self.mixup > 0.0:
            c = self.mixup_criterion(criteron, logits, y_a, y_b, mixup_l)
        else:
            c = criteron(logits, y)
            # c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

        # Accumulate grads
        # with self.chrono.measure("grads"):
        #     loss = (c / self.hparams.batch_split)

        # step = self.global_step // self.hparams.batch_split
        # accstep = f" ({self.global_step - step}/{self.hparams.batch_split})" if self.hparams.batch_split > 1 else ""
        # lr = bit_hyperrule.get_lr(
        #     step, base_lr=self.hparams.base_lr, supports=self.lr_supports)
        # self.my_logger.info(
        #     f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
        # self.my_logger.flush()

        acc1, acc5 = self.__accuracy(logits, y, topk=(1, 5))

        tqdm_dict = {'train_loss': c}
        output = OrderedDict({
            'loss': c,
            'acc1': acc1,
            'acc5': acc5,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    # def on_train_end(self):
    #     self.my_logger.info(f"Timings:\n{self.chrono}")

    def validation_step(self, batch, batch_idx):
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

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}

        all_size = torch.stack([o['batch_size'] for o in outputs]).sum()
        for metric_name in ["val_loss", "val_acc1", "val_acc5"]:
            metrics = [o[metric_name] for o in outputs]
            tqdm_dict[metric_name] = torch.sum(torch.stack(metrics)) / all_size

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"],
            'step': self.global_step, # for checkpoint filename
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
            # subtle: it only calls scheudler every {num_GPU} steps
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: bit_hyperrule.get_lr(
                    self.global_step, base_lr=self.hparams.base_lr, supports=self.lr_supports)),
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

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                    second_order_closure=None):
    #     # Update learning-rate, including stop training if over.
    #     step = self.global_step // self.hparams.batch_split
    #     lr = bit_hyperrule.get_lr(
    #         step, base_lr=self.hparams.base_lr, supports=self.lr_supports)
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = lr
    #
    #     if self.global_step - step == (self.hparams.batch_split - 1):
    #         with self.chrono.measure("update"):
    #             optimizer.step()
    #             optimizer.zero_grad()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

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

    def mktrainval(self):
        if self.hparams.dataset not in ['objectnet', 'imageneta'] or self.hparams.inpaint == 'none':
            return self._mktrainval()
        else:
            self.my_logger.info(f"Composing 2 loaders for {self.hparams.dataset} w/ inpaint {self.hparams.inpaint}")
            # Compose 2 loaders: 1 w/ inpaint as true and dataset == 'objectnet_bbox'
            # The other would be having 1 w/ inpaint == 'none' and dataset == 'objectnet_no_bbox'
            orig_inpaint = self.hparams.inpaint
            orig_dataset = self.hparams.dataset

            self.hparams.dataset = f'{orig_dataset}_bbox'
            f_n_train, n_classes, f_train_loader, valid_loader = \
                self.mktrainval()
            self.hparams.dataset = f'{orig_dataset}_no_bbox'
            self.hparams.inpaint = 'none'
            s_n_train, _, s_train_loader, _ = self.mktrainval()

            n_train = f_n_train + s_n_train

            def composed_train_loader():
                loaders = [f_train_loader, s_train_loader]
                order = np.random.randint(low=0, high=2)
                for s in loaders[order]:
                    yield s
                self.my_logger.info(f"Finish the {order} loader. (0 means bbox, 1 means no bbox)")
                for s in loaders[1 - order]:
                    yield s
                self.my_logger.info(f"Finish the {1 - order} loader. (0 means bbox, 1 means no bbox)")

            train_loader = composed_train_loader()

            # Set everything back
            self.hparams.dataset = orig_dataset
            self.hparams.inpaint = orig_inpaint
            self.my_logger.info(f"Using a total training set {n_train} images")
            return n_train, n_classes, train_loader, valid_loader

    def _mktrainval(self):
        """Returns train and validation datasets."""
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

        collate_fn = None
        n_train = None
        micro_batch_size = self.hparams.batch // self.hparams.batch_split
        if self.use_ddp: # see
            micro_batch_size //= (torch.cuda.device_count())

        if self.hparams.dataset == "cifar10":
            train_set = tv.datasets.CIFAR10(self.hparams.datadir, transform=train_tx, train=True, download=True)
            valid_set = tv.datasets.CIFAR10(self.hparams.datadir, transform=val_tx, train=False, download=True)
        elif self.hparams.dataset == "cifar100":
            train_set = tv.datasets.CIFAR100(self.hparams.datadir, transform=train_tx, train=True, download=True)
            valid_set = tv.datasets.CIFAR100(self.hparams.datadir, transform=val_tx, train=False, download=True)
        elif self.hparams.dataset == "imagenet2012":
            train_set = MyImageFolder(pjoin(self.hparams.datadir, "train"), transform=train_tx)
            valid_set = MyImageFolder(pjoin(self.hparams.datadir, "val"), transform=val_tx)
        elif self.hparams.dataset.startswith('objectnet') or self.hparams.dataset.startswith(
                'imageneta'):  # objectnet and objectnet_bbox and objectnet_no_bbox
            identifier = 'objectnet' if self.hparams.dataset.startswith('objectnet') else 'imageneta'
            valid_set = MyImageFolder(f"../datasets/{identifier}/", transform=val_tx)

            if self.hparams.inpaint == 'none':
                if self.hparams.dataset == 'objectnet' or self.hparams.dataset == 'imageneta':
                    train_set = MyImageFolder(pjoin(self.hparams.datadir, f"train_{self.hparams.dataset}"),
                                                        transform=train_tx)
                else:  # For only images with or w/o bounding box
                    train_bbox_file = '../datasets/imagenet/LOC_train_solution_size.csv'
                    df = pd.read_csv(train_bbox_file)
                    filenames = set(df[df.bbox_ratio <= self.hparams.bbox_max_ratio].ImageId)
                    if self.hparams.dataset == f"{identifier}_no_bbox":
                        is_valid_file = lambda path: os.path.basename(path).split('.')[0] not in filenames
                    elif self.hparams.dataset == f"{identifier}_bbox":
                        is_valid_file = lambda path: os.path.basename(path).split('.')[0] in filenames
                    else:
                        raise NotImplementedError()

                    train_set = MyImageFolder(
                        pjoin(self.hparams.datadir, f"train_{identifier}"),
                        is_valid_file=is_valid_file,
                        transform=train_tx)
            else:  # do inpainting
                train_tx = tv.transforms.Compose([
                    data_utils.Resize((precrop, precrop)),
                    data_utils.RandomCrop((crop, crop)),
                    data_utils.RandomHorizontalFlip(),
                    data_utils.ToTensor(),
                    data_utils.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

                train_set = MyImagenetBoundingBoxFolder(
                    root=f"../datasets/imagenet/train_{identifier}",
                    bbox_file='../datasets/imagenet/LOC_train_solution.csv',
                    transform=train_tx)
                collate_fn = bbox_collate
                n_train = len(train_set) * 2
                micro_batch_size //= 2
        else:
            raise ValueError(f"Sorry, we have not spent time implementing the "
                             f"{self.hparams.dataset} dataset in the PyTorch codebase. "
                             f"In principle, it should be easy to add :)")

        if self.hparams.examples_per_class is not None:
            self.my_logger.info(f"Looking for {self.hparams.examples_per_class} images per class...")
            indices = fs.find_fewshot_indices(train_set, self.hparams.examples_per_class)
            train_set = torch.utils.data.Subset(train_set, indices=indices)

        self.my_logger.info(f"Using a training set with {len(train_set)} images.")
        self.my_logger.info(f"Using a validation set with {len(valid_set)} images.")

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=micro_batch_size, shuffle=False,
            num_workers=self.hparams.workers, pin_memory=True, drop_last=False)

        if n_train is None:
            n_train = len(train_set)
        self.lr_supports = bit_hyperrule.get_schedule(n_train)
        ## hack to make the pl train for 1 epoch
        train_set.num_samples = (self.hparams.batch * self.lr_supports[-1])

        if micro_batch_size <= len(train_set):
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=micro_batch_size, shuffle=True,
                num_workers=self.hparams.workers, pin_memory=True, drop_last=False,
                collate_fn=collate_fn)
        else:
            # In the few-shot cases, the total dataset size might be smaller than the batch-size.
            # In these cases, the default sampler doesn't repeat, so we need to make it do that
            # if we want to match the behaviour from the paper.
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=micro_batch_size, num_workers=self.hparams.workers,
                pin_memory=True,
                sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size),
                collate_fn=collate_fn)

        return n_train, len(valid_set.classes), train_loader, valid_loader

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser
        # parser = ArgumentParser(parents=[parent_parser])
        # return parser

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

    # def my_load_model(self):
    #     try:
    #         logger.info(f"Model will be saved in '{savename}'")
    #         checkpoint = torch.load(savename, map_location="cpu")
    #         logger.info(f"Found saved model to resume from at '{savename}'")
    #
    #         step = checkpoint["step"]
    #         model.load_state_dict(checkpoint["model"])
    #         model = model.to(device)
    #
    #         # Note: no weight-decay!
    #         optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    #         optim.load_state_dict(checkpoint["optim"])
    #         logger.info(f"Resumed at step {step}")
    #     except FileNotFoundError:
    #         if args.finetune:
    #             logger.info("Fine-tuning from BiT")
    #             model.load_from(np.load(f"models/{args.model}.npz"))
    #
    #         model = model.to(device)
    #         optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)


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
    parser.add_argument("--dataset", choices=list(bit_hyperrule.known_dataset_sizes.keys()),
                        default='objectnet_bbox',
                        help="Choose the dataset. It should be easy to add your own! "
                             "Don't forget to set --datadir if necessary.")
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

    parser.add_argument("--datadir", default='../datasets/imagenet',
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")

    # My own arguments
    parser.add_argument("--inpaint", type=str, default='none',
                        choices=['mean', 'random', 'local',
                                 'cagan', 'none'])
    parser.add_argument("--bbox_subsample_ratio", type=float, default=1)
    parser.add_argument("--bbox_max_ratio", type=float, default=0.5)
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
        filepath=pjoin(args.logdir, args.name, '{step}'),
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
