import torch
import torch.nn.functional as F
from os.path import join as pjoin
from os.path import exists as pexists
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import resnet50
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.calibration import calibration_curve
import numpy as np

from .xray import XRayLightningModel
from ..data.cct_datasets import MyCCT_Dataset
from ..data.imagenet_datasets import MyImageFolder, MyImagenetBoundingBoxFolder
from ..data import bbox_utils
import torchvision as tv
from .. import models
from ..data.imagenet_datasets import MySubset, MyConcatDataset
from ..saliency_utils import get_grad_y, get_grad_sum, \
    get_grad_logp_sum, get_grad_logp_y, get_deeplift


class IN9LightningModel(XRayLightningModel):
    def init_setup(self):
        # Resnet 50
        arch = 'BiT-S-R50x1'
        if 'arch' in self.hparams:
            arch = self.hparams.arch

        self.model = models.KNOWN_MODELS[arch](
            head_size=9,
            zero_head=False)
        # self.my_logger.info("Fine-tuning from BiT")
        # self.model.load_from(np.load(f"models/BiT-S-R50x1.npz"))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        logit = self(x)
        prefix = ['val', 'orig', 'mixed_same', 'mixed_rand',
                  'mixed_next'][dataloader_idx]
        output = OrderedDict({
            f'{prefix}_logit': logit,
            f'{prefix}_y': y,
        })
        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        calibration_dict = {}

        def cal_metrics(output, prefix='val'):
            logit = torch.cat([o[f'{prefix}_logit'] for o in output])
            y = torch.cat([o[f'{prefix}_y'] for o in output])
            tqdm_dict[f'{prefix}_loss'] = F.cross_entropy(logit, y, reduction='mean').item()
            tqdm_dict[f'{prefix}_acc1'], = self.accuracy(logit, y, topk=(1,))

            # Calculate the average auc, average aupr, and average F1
            _, pred = torch.max(logit, dim=1)
            y_onehot = torch.nn.functional.one_hot(y, num_classes=logit.shape[1])

            prob = F.softmax(logit, dim=1)
            y, y_onehot, logit, pred = y.cpu().numpy(), y_onehot.cpu().numpy(), \
                                       logit.cpu().numpy(), pred.cpu().numpy()

            all_y = y_onehot.reshape(-1)
            all_prob = prob.reshape(-1).cpu().numpy()

            fraction_of_positives, mean_predicted_value = \
                calibration_curve(all_y, all_prob, n_bins=10)
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))

            tqdm_dict[f'{prefix}_ece'] = ece * 100
            # tqdm_dict[f'{prefix}_f1'] = f1_score(
            #     y, pred, average='macro') * 100
            # tqdm_dict[f'{prefix}_auc'] = roc_auc_score(
            #     y_onehot, logit, multi_class='ovr') * 100
            # tqdm_dict[f'{prefix}_aupr'] = average_precision_score(
            #     y_onehot, logit) * 100

            hist, bins = np.histogram(all_prob, bins=10)
            calibration_dict[f'{prefix}_frp'] = fraction_of_positives.tolist()
            calibration_dict[f'{prefix}_mpv'] = mean_predicted_value.tolist()
            calibration_dict[f'{prefix}_hist'] = hist.tolist()
            calibration_dict[f'{prefix}_bins'] = bins.tolist()

        cal_metrics(outputs[0], 'val')
        cal_metrics(outputs[1], 'orig')
        cal_metrics(outputs[2], 'mixed_same')
        cal_metrics(outputs[3], 'mixed_rand')
        cal_metrics(outputs[4], 'mixed_next')

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict,
            'val_loss': tqdm_dict["val_loss"],
            **calibration_dict
        }
        return result

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(), lr=self.hparams.base_lr,
            momentum=0.9, weight_decay=1e-4)
        scheduler = {
            # Total 50 epochs
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(
                optim, milestones=[6, 12, 18], gamma=0.1),
            'interval': 'epoch',
        }
        return [optim], [scheduler]

    def _make_train_val_dataset(self):
        train_d = MyImagenetBoundingBoxFolder(
            '../datasets/bg_challenge/train/original/train/',
            '../datasets/imagenet/LOC_train_solution.csv',
            transform=MyImagenetBoundingBoxFolder.get_train_transform(
                self.hparams.test_run))
        val_d = MyImagenetBoundingBoxFolder(
            '../datasets/bg_challenge/train/original/val/',
            '../datasets/imagenet/LOC_train_solution.csv',
            transform=MyImagenetBoundingBoxFolder.get_val_transform(
                self.hparams.test_run))
        orig_test_d = MyImagenetBoundingBoxFolder(
            '../datasets/bg_challenge/test/original/val/',
            '../datasets/imagenet/LOC_val_solution.csv',
            transform=MyImagenetBoundingBoxFolder.get_val_transform(
                self.hparams.test_run))
        mixed_same_test_d = MyImagenetBoundingBoxFolder(
            '../datasets/bg_challenge/test/mixed_same/val/',
            '../datasets/imagenet/LOC_val_solution.csv',
            transform=MyImagenetBoundingBoxFolder.get_val_transform(
                self.hparams.test_run))
        mixed_rand_test_d = MyImagenetBoundingBoxFolder(
            '../datasets/bg_challenge/test/mixed_rand/val/',
            '../datasets/imagenet/LOC_val_solution.csv',
            transform=MyImagenetBoundingBoxFolder.get_val_transform(
                self.hparams.test_run))
        mixed_next_test_d = MyImagenetBoundingBoxFolder(
            '../datasets/bg_challenge/test/mixed_next/val/',
            '../datasets/imagenet/LOC_val_solution.csv',
            transform=MyImagenetBoundingBoxFolder.get_val_transform(
                self.hparams.test_run))

        return train_d, [val_d, orig_test_d, mixed_same_test_d,
                         mixed_rand_test_d, mixed_next_test_d]

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--arch", type=str, default='BiT-S-R50x1')
        parser.add_argument("--max_epochs", type=int, default=25)
        parser.add_argument("--batch", type=int, default=32,
                            help="Batch size.")
        parser.add_argument("--val_batch", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--batch_split", type=int, default=1,
                            help="Number of batches to compute gradient on before updating weights.")
        parser.add_argument("--base_lr", type=float, default=0.05)
        parser.add_argument("--pl_model", type=str, default=cls.__name__)
        parser.add_argument("--reg_anneal", type=float, default=0.)
        return parser

    def pl_trainer_args(self):
        checkpoint_callback = ModelCheckpoint(
            filepath=pjoin(self.hparams.logdir, self.hparams.name, '{epoch}'),
            save_top_k=1,
            save_last=True,
            verbose=True,
            mode='max',
            monitor='val_acc1',
        )

        args = dict()
        args['max_epochs'] = self.hparams.max_epochs
        args['checkpoint_callback'] = checkpoint_callback

        last_ckpt = pjoin(self.hparams.logdir, self.hparams.name, 'last.ckpt')
        if pexists(last_ckpt):
            args['resume_from_checkpoint'] = last_ckpt
        return args

    def get_grad_cam_layer(self):
        return self.model.head[1]
