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
from os.path import exists as pexists
import json

import argparse
from argparse import ArgumentParser, Namespace
import os
import random
import numpy as np
import shutil
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.logging import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger
from pytorch_lightning import Trainer, seed_everything

from arch.lightning.csv_recording import CSVRecordingCallback
from arch.lightning.base_imagenet import ImageNetLightningModel
from arch.lightning.xray import XRayLightningModel
from arch.lightning.cct import CCTLightningModel
from arch.lightning.in9 import IN9LightningModel, IN9LLightningModel
from arch.utils import Timer


def get_args():
    # Big Transfer arg parser
    parser = ArgumentParser(description="Fine-tune BiT-M model.")
    parser.add_argument('--distributed_backend', type=str, default='dp',
                        choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument("--name",
                        default='test',
                        # default='0930_in9_seg_random_s5',
                        help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument("--model", default='BiT-S-R50x1',
                        help="Which variant to use; BiT-M gives best results.")
    # parser.add_argument("--logdir", default='./models/',
    #                     help="Where to log training info (small).")
    parser.add_argument('--seed', type=int, default=123,
                        help='seed for initializing training. ')
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    # My own arguments
    parser.add_argument("--dataset", default='in9',
                        help="Choose the dataset. It should be easy to add your own! "
                             "Don't forget to set --datadir if necessary.",
                        choices=['objectnet', 'imageneta', 'xray', 'cct', 'in9', 'in9l'])
    parser.add_argument("--datadir", default='../datasets/imagenet',
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--inpaint", type=str, default='none')  # ['none', 'random', 'mean', 'shuffle', 'tile']
    parser.add_argument("--f_inpaint", type=str, default='none')  # ['none', 'pgd', 'fgsm', 'mean', 'random']
    parser.add_argument("--alpha", type=float, default=-1)  # For factual inpainting
    parser.add_argument("--eps", type=float, default=0.6)  # For factual inpainting
    parser.add_argument("--reg", type=str, default='none') # ['none', 'gs', 'bbox_o']
    parser.add_argument("--reg_grad", type=str, default='grad_y',
                        choices=['grad_y', 'grad_sum', 'grad_logp_y',
                                 'grad_logp_sum', 'deeplift', 'grad_cam'])
    parser.add_argument("--reg_coeff", type=float, default=0.)
    parser.add_argument("--cf", type=str, default='logp') # ['logp', 'uni', 'uni_e']
    parser.add_argument("--cf_coeff", type=float, default=1.)
    # parser.add_argument("--2xchannels", type=int, default=0)
    # parser.add_argument("--use_mixup", type=int, default=0)
    parser.add_argument("--test_run", type=int, default=1)
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--finetune", type=int, default=1)

    temp_args, _ = parser.parse_known_args()

    # setup which lightning model to use
    pl_model_dict = {
        'objectnet': ImageNetLightningModel.__name__,
        'imageneta': ImageNetLightningModel.__name__,
        'xray': XRayLightningModel.__name__,
        'cct': CCTLightningModel.__name__,
        'in9': IN9LightningModel.__name__,
        'in9l': IN9LLightningModel.__name__,
    }
    temp_args.pl_model = pl_model_dict[temp_args.dataset]
    parser = eval(temp_args.pl_model).add_model_specific_args(parser)

    saved_hparams = pjoin('models', temp_args.name, 'hparams.json')
    if not pexists(saved_hparams):
        args = parser.parse_args()
    else:
        hparams = json.load(open(saved_hparams))

        # Remove default value. Only parse user inputs
        for action in parser._actions:
            action.default = argparse.SUPPRESS
        input_args = parser.parse_args()
        print('Reload and update from inputs: ' + str(input_args))
        if len(vars(input_args)) > 0:
            hparams.update(vars(input_args))
            json.dump(hparams, open(saved_hparams, 'w'))
        args = Namespace(**hparams)

    args.logdir = './models/'
    args.result_dir = './results/'
    os.makedirs(args.result_dir, exist_ok=True)

    # on v server
    if not pexists(pjoin(args.logdir, args.name)) \
        and 'SLURM_JOB_ID' in os.environ \
        and pexists('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID']):
        os.symlink('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID'],
                   pjoin(args.logdir, args.name))

    if args.test_run:
        print("WATCHOUT!!! YOU ARE RUNNING IN TEST RUN MODE!!!")

        args.batch = 8
        args.val_batch = 16
        args.batch_split = 2
        args.workers = 0
        # args.eval_every = 1000
        args.inpaint = 'none'
        # args.inpaint = 'cagan'
        # args.f_inpaint = 'none'
        args.f_inpaint = 'none'
        args.reg = 'none'
        # args.reg = 'bbox_f1'
        args.reg_coeff = 1e-5
        # args.reg_grad = 'grad_cam'
        # args.reg_grad = 'deeplift'
        args.reg_anneal = 0.
        args.fp16 = 0
        args.max_epochs = 5
        # args.cf = 'uni_e'
        args.eval_every = 100
        if args.dataset.startswith('in9'):
            args.mask = 'seg'
            args.data_ratio = 1.2
        if args.dataset.startswith('imageneta'):
            args.nobbox_data = 1.

        if pexists('./models/test'):
            shutil.rmtree('./models/test', ignore_errors=True)
        if pexists('./lightning_logs/test'):
            shutil.rmtree('./lightning_logs/test', ignore_errors=True)

    if args.reg == 'grad_supervision':
        assert args.inpaint != 'none', 'wrong argument!'
    if args.reg == 'grad_cam':
        assert torch.cuda.device_count() == 1, 'Now grad cam does not allow multi gpus'
    return args


def main(args: Namespace) -> None:
    # model = eval(args.pl_model).init_model(args)
    model = eval(args.pl_model)(args)

    if args.seed is not None:
        seed_everything(args.seed)

    os.makedirs(pjoin('./lightning_logs', args.name), exist_ok=True)
    logger = TensorBoardLogger(
        save_dir='./lightning_logs/',
        name=args.name,
    )

    default_args = dict(
        gpus=-1,
        distributed_backend=args.distributed_backend,
        precision=16 if args.fp16 else 32,
        amp_level='O1',
        profiler=True,
        num_sanity_val_steps=1,
        # num_sanity_val_steps=1 if args.test_run else 0,
        accumulate_grad_batches=args.batch_split,
        logger=logger,
        benchmark=(args.seed is None),
        deterministic=(args.seed is not None),
        callbacks=[CSVRecordingCallback()],
        limit_val_batches=0.1 if args.test_run else 1.,
        limit_train_batches=0.1 if args.test_run else 1.,
        gradient_clip_val=1.0,
    )
    # Let the specific model could overwrite the default args
    default_args.update(model.pl_trainer_args())

    trainer = pl.Trainer(**default_args)
    if not model.is_finished_run(pjoin(args.logdir, args.name)):
        # Record hyperparameters
        json.dump(vars(args),
                  open(pjoin(args.logdir, args.name, 'hparams.json'), 'w'))

        trainer.fit(model)

    # Run the test set if defined in lightning model
    gave_test_loader = hasattr(model, 'test_dataloader') and \
                       model.test_dataloader()
    if gave_test_loader:
        with Timer('testing'):
            trainer.test(model)


if __name__ == '__main__':
    main(get_args())
