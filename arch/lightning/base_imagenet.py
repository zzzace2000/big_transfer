import os
from collections import OrderedDict
from os.path import join as pjoin  # pylint: disable=g-importing-member
from os.path import exists as pexists
import sys

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
from sklearn.metrics import average_precision_score
from pytorch_lightning.callbacks import ModelCheckpoint

from .. import models, bit_hyperrule
from ..data import bbox_utils
from ..data.simulated_datasets import GaussianNoiseDataset, UniformNoiseDataset
from ..data.xrayvision_datasets import Kaggle_Dataset
from ..data.cct_datasets import MyCCT_Dataset
from ..data.imagenet_datasets import MyImagenetBoundingBoxFolder, MyImageFolder, MyConcatDataset, MySubset, \
    MyImageNetODataset, MyFactualAndCFDataset
from ..inpainting.VAEInpainter import VAEInpainter
from ..utils import output_csv, DotDict, generate_mask
from ..saliency_utils import get_grad_y, get_grad_sum, \
    get_grad_logp_sum, get_grad_logp_y, get_deeplift, get_grad_cam
from .base import BaseLightningModel


KNOWN_NUM_CLASSES = {
    'imageneta': 200,
    'objectnet': 113,
}


class ImageNetLightningModel(BaseLightningModel):
    def init_setup(self):
        if 'lr_steps' not in self.hparams:
            self.hparams.lr_steps = 10000

        # setup the learning rate schedule for how long we want to train
        self.lr_supports = [
            int(r * self.hparams.lr_steps) for r in [0.05, 0.3, 0.6, 0.9, 1.0]
        ]

        if self.hparams.model in models.KNOWN_MODELS:
            self.model = models.KNOWN_MODELS[self.hparams.model](
                head_size=KNOWN_NUM_CLASSES[self.hparams.dataset],
                zero_head=self.hparams.finetune)
            if self.hparams.finetune:
                self.my_logger.info("Fine-tuning from BiT")
                self.model.load_from(np.load(f"models/{self.hparams.model}.npz"))
        else:  # from torchvision
            raise NotImplementedError()

        self.mixup = 0
        # if hparams.use_mixup:
        #     self.mixup = bit_hyperrule.get_mixup(len(train_set))

    def on_fit_start(self):
        if self.global_step >= self.hparams.lr_steps:
            self.my_logger.info('Already finish training! Exit!')
            sys.exit()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # if dataloader_idx in [0, 1]:
        images, target = batch
        if isinstance(images, dict):
            images = images['imgs']

        output = self(images)
        loss_val = F.cross_entropy(output, target, reduction='sum')
        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        batch_size = images.new_tensor(images.shape[0])

        output = OrderedDict({
            'loss': loss_val,
            'acc1': acc1 * batch_size,
            'acc5': acc5 * batch_size,
            'batch_size': batch_size,
        })
        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}

        def cal_output(the_outputs, the_prefix):
            all_size = torch.stack([o['batch_size'] for o in the_outputs]).sum()
            for metric_name in ["loss", "acc1", "acc5"]:
                metrics = [o[metric_name] for o in the_outputs]
                tqdm_dict['%s_%s' % (the_prefix, metric_name)] = \
                    torch.sum(torch.stack(metrics)) / all_size

        # 1st val loader
        cal_output(outputs[0], 'val')
        cal_output(outputs[1], 'test')

        # 3rd val loader
        # if isinstance(outputs[0], list) and len(outputs) > 2:
        #     the_outputs = outputs[2]
        #     anomaly_scores = torch.cat([o['as'] for o in the_outputs])
        #     ys = torch.cat([o['y'] for o in the_outputs])
        #
        #     tqdm_dict['imgneto_aupr'] = average_precision_score(
        #         ys.cpu().numpy(), anomaly_scores.cpu().numpy())

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict,
            'val_loss': tqdm_dict["val_loss"],
            'gstep': self.global_step, # checkpoint filename
        }
        return result

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

    def _make_train_val_dataset(self):
        df = pd.read_csv(pjoin('../datasets/imagenet/', 'LOC_train_solution_size.csv'))
        def has_bbox(inverse=False):
            bbox_filenames = set(df[df.bbox_ratio <= self.hparams.bbox_max_ratio].ImageId)
            def is_valid_file(path):
                ans = os.path.basename(path).split('.')[0] in bbox_filenames
                if inverse:
                    ans = (not ans)
                return ans
            return is_valid_file

        train_sets = []

        # handle bbox dataset
        assert self.hparams.bbox_data > 0.
        if self.hparams.inpaint == 'none' and self.hparams.reg == 'none' \
                and self.hparams.f_inpaint == 'none':
            bbox_d = MyImageFolder(
                pjoin('../datasets/imagenet/', f"train_{self.hparams.dataset}"),
                is_valid_file=has_bbox(),
                transform=MyImageFolder.get_train_transform(self.hparams.test_run))
        else:
            cf_inpaint_dir = None
            if self.hparams.inpaint == 'cagan':
                cf_inpaint_dir = \
                    pjoin('../datasets/imagenet/', f"train_cagan_{self.hparams.dataset}")
            bbox_d = MyImagenetBoundingBoxFolder(
                pjoin('../datasets/imagenet/', f"train_{self.hparams.dataset}"),
                bbox_file=pjoin('../datasets/imagenet/', 'LOC_train_solution.csv'),
                is_valid_file=has_bbox(),
                cf_inpaint_dir=cf_inpaint_dir,
                transform=MyImagenetBoundingBoxFolder.get_train_transform(
                    self.hparams.test_run))

        num = int(len(bbox_d) * self.hparams.bbox_data)
        bbox_d, _ = self.sub_dataset(bbox_d, num)
        self.my_logger.info(f"W/ bbox total examples are {len(bbox_d)}")
        train_sets.append(bbox_d)

        # handle no bbox dataset
        if self.hparams.nobbox_data > 0. or self.hparams.nobbox_data == -1:
            nobbox_d = MyImageFolder(
                pjoin('../datasets/imagenet/', f"train_{self.hparams.dataset}"),
                is_valid_file=has_bbox(inverse=True),
                transform=MyImageFolder.get_train_transform(self.hparams.test_run))

            if self.hparams.nobbox_data > 0.:
                num_nobbox_data = int(self.hparams.nobbox_data * len(bbox_d))
                nobbox_d, _ = self.sub_dataset(nobbox_d, num_nobbox_data)
            self.my_logger.info(f"W/o bbox total examples are {len(nobbox_d)}")
            train_sets.append(nobbox_d)

        train_set = train_sets[0] if len(train_sets) == 1 \
            else MyConcatDataset(train_sets)

        ## Validation set
        val_tx = MyImageFolder.get_val_transform(self.hparams.test_run)
        valid_sets = [
            MyImageFolder(f"../datasets/imagenet/val_{self.hparams.dataset}/",
                          transform=val_tx), # Orig test set
            MyImageFolder(f"../datasets/{self.hparams.dataset}/",
                          transform=val_tx), # Imagenet-a
        ]

        # Hack to make the pl train for 1 epoch = this number of steps.
        train_set.my_num_samples = \
            (self.hparams.batch * (self.lr_supports[-1]))
        return train_set, valid_sets

    def get_inpainting_model(self, inpaint):
        if inpaint == 'cagan':
            return None

        if inpaint == 'vae':
            inpaint_model = VAEInpainter(in_mean=(0.5, 0.5, 0.5), in_std=(0.5, 0.5, 0.5))
            gen_model_path = './inpainting_models/0928-VAE-Var-hole_lr_0.0002_epochs_7'
            inpaint_model.load_state_dict(
                torch.load(gen_model_path, map_location=lambda storage, loc: storage)['state_dict'],
                strict=False)
            return inpaint_model

        return super().get_inpainting_model(inpaint)

    def _make_test_datasets(self):
        orig_test_d = MyImageFolder(
            f"../datasets/imagenet/val_{self.hparams.dataset}/",
            transform=MyImageFolder.get_val_transform(self.hparams.test_run))

        gn_d = GaussianNoiseDataset(num_samples=len(orig_test_d))
        un_d = UniformNoiseDataset(num_samples=len(orig_test_d))
        cct_d = MyCCT_Dataset(
            '../datasets/cct/CaltechCameraTrapsECCV18.json',
            transform=MyCCT_Dataset.get_val_bbox_transform()
        )

        xray_tx = Kaggle_Dataset.get_val_transform(self.hparams.test_run)
        # Remove grey scale to keep 3 channels
        xray_tx.transforms = xray_tx.transforms[1:]
        xray_d = Kaggle_Dataset(
            f"../datasets/kaggle/stage_2_train_images_jpg/",
            include='all',
            transform=xray_tx)

        imageneto_d = MyImageFolder(
            '../datasets/imageneto/',
            transform=MyImageFolder.get_val_transform(self.hparams.test_run))

        test_sets = [orig_test_d, gn_d, un_d, cct_d, xray_d, imageneto_d]
        self.test_sets_names = ['orig', 'gn', 'un', 'cct', 'xray', 'imageneto']
        return test_sets

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Override to calculate the original test accuacy to fix previous bugs
        '''
        if dataloader_idx != 0:
            return super().test_step(batch, batch_idx, dataloader_idx)

        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        logits = self(x)
        prob = F.softmax(logits, dim=1)
        anomaly_score = 1. - (prob.max(dim=1).values)
        # Anomaly score...

        acc1, acc5 = self.accuracy(logits, y, topk=(1, 5))
        batch_size = x.new_tensor(x.shape[0])

        output = {
            'anomaly_score': anomaly_score,
            'acc1': acc1 * batch_size,
            'acc5': acc5 * batch_size,
            'batch_size': batch_size,
        }
        return output

    def test_epoch_end(self, outputs):
        result = super().test_epoch_end(outputs)

        tqdm_dict = result['log']

        orig_output = outputs[0]
        all_size = torch.stack([o['batch_size'] for o in orig_output]).sum()
        for metric_name in ["acc1", "acc5"]:
            metrics = [o[metric_name] for o in orig_output]
            tqdm_dict['orig_test_%s' % metric_name] = \
                torch.sum(torch.stack(metrics)) / all_size

        return result

    def is_data_ratio_exp(self):
        return self.hparams.bbox_data != 1. or self.hparams.nobbox_data != 0. \
            or 'bd' in self.hparams.name

    @classmethod
    def add_model_specific_args(cls, parser):  # pragma: no-cover
        parser.add_argument("--bbox_data", type=float, default=1.)
        parser.add_argument("--bbox_max_ratio", type=float, default=0.5)
        parser.add_argument("--nobbox_data", type=float, default=0.)

        parser.add_argument("--batch", type=int, default=32,
                            help="Batch size.")
        parser.add_argument("--val_batch", type=int, default=512,
                            help="Batch size.")
        parser.add_argument("--batch_split", type=int, default=1,
                            help="Number of batches to compute gradient on before updating weights.")
        parser.add_argument("--base_lr", type=float, default=0.003,
                            help="Base learning-rate for fine-tuning. Most likely default is best.")
        parser.add_argument("--eval_every", type=int, default=2000,
                            help="Run prediction on validation set every so many steps."
                                 "Will always run one evaluation at the end of training.")
        # parser.add_argument("--val_data", type=float, default=2000)
        # parser.add_argument("--test_data", type=float, default=2000)
        parser.add_argument("--pl_model", type=str, default=cls.__name__)
        parser.add_argument("--reg_anneal", type=float, default=0.)
        parser.add_argument("--lr_steps", type=int, default=20000)
        return parser

    def pl_trainer_args(self):
        checkpoint_callback = ModelCheckpoint(
            filepath=pjoin(self.hparams.logdir, self.hparams.name, '{gstep}'),
            save_top_k=1,
            save_last=True,
            period=-1, # -1 then it saves checkpts within an epoch
            verbose=True,
            monitor='val_acc1',
            mode='max',
        )

        val_check_interval = self.hparams.eval_every
        if self.hparams.distributed_backend == 'dp':
            val_check_interval *= self.hparams.batch_split

        args = dict()
        args['max_steps'] = self.hparams.lr_steps
        args['val_check_interval'] = val_check_interval
        args['checkpoint_callback'] = checkpoint_callback
        last_ckpt = pjoin(self.hparams.logdir, self.hparams.name, 'last.ckpt')
        if pexists(last_ckpt):
            args['resume_from_checkpoint'] = last_ckpt

        return args

    @classmethod
    def is_finished_run(cls, model_dir):
        last_ckpt = pjoin(model_dir, 'last.ckpt')
        if pexists(last_ckpt):
            tmp = torch.load(last_ckpt,
                             map_location=torch.device('cpu'))
            last_gstep = tmp['global_step']
            hparams = tmp['hyper_parameters']
            if last_gstep >= (hparams.get('lr_steps', 10000) - 1):
                print('Already finish fitting! Max %d Last %d'
                      % (hparams.get('lr_steps', 10000), last_gstep))
                return True

        return False
