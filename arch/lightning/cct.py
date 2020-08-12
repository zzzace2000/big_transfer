import torch
import torch.nn.functional as F
from os.path import join as pjoin
from os.path import exists as pexists
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import resnet50
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from .xray import XRayLightningModel
from ..data.cct_datasets import MyCCT_Dataset

from ..data.imagenet_datasets import MySubset, MyConcatDataset
from ..saliency_utils import get_grad_y, get_grad_sum, \
    get_grad_logp_sum, get_grad_logp_y, get_deeplift


class CCTLightningModel(XRayLightningModel):
    def init_setup(self):
        # Resnet 50
        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, 16, bias=True)
        torch.nn.init.zeros_(self.model.fc.weight)
        torch.nn.init.zeros_(self.model.fc.bias)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        logit = self(x)
        prefix = ['val', 'cis_test', 'trans_test'][dataloader_idx]

        output = OrderedDict({
            f'{prefix}_logit': logit,
            f'{prefix}_y': y,
        })
        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}

        def cal_metrics(output, prefix='val'):
            logit = torch.cat([o[f'{prefix}_logit'] for o in output])
            y = torch.cat([o[f'{prefix}_y'] for o in output])
            tqdm_dict[f'{prefix}_loss'] = F.cross_entropy(logit, y, reduction='mean').item()
            tqdm_dict[f'{prefix}_acc1'], = self.accuracy(logit, y, topk=(1,))

            # Calculate the average auc, average aupr, and average F1
            _, pred = torch.max(logit, dim=1)
            y_onehot = torch.nn.functional.one_hot(y, num_classes=16)

            # In cis_val and trans_test, some classes do not exist
            non_zero_cls = (y_onehot.sum(dim=0) > 0)
            if not non_zero_cls.all():
                y_onehot = y_onehot[:, non_zero_cls]
                logit = logit[:, non_zero_cls]

            y, y_onehot, logit, pred = y.cpu().numpy(), y_onehot.cpu().numpy(), \
                                      logit.cpu().numpy(), pred.cpu().numpy()
            tqdm_dict[f'{prefix}_f1'] = f1_score(
                y, pred, average='macro') * 100
            tqdm_dict[f'{prefix}_auc'] = roc_auc_score(
                y_onehot, logit, multi_class='ovr') * 100
            tqdm_dict[f'{prefix}_aupr'] = average_precision_score(
                y_onehot, logit) * 100

        cal_metrics(outputs[0], 'val')
        cal_metrics(outputs[1], 'cis_test')
        cal_metrics(outputs[2], 'trans_test')

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict,
            'val_loss': tqdm_dict["val_loss"],
        }
        return result

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.model.parameters(), lr=self.hparams.base_lr,
            momentum=0.9)
        return optim

    @classmethod
    def counterfactual_ce_loss(cls, logit, y):
        '''
        If it's counterfactual, assign it to empty class 11
        '''
        assert (y < 0).all(), str(y)
        cf_y = 11 * torch.ones_like(y).long()
        return F.cross_entropy(logit, cf_y, reduction='mean')

    def _make_train_val_dataset(self):
        train_d = MyCCT_Dataset(
            '../datasets/cct/eccv_18_annotation_files/train_annotations.json',
            transform=MyCCT_Dataset.get_train_bbox_transform()
        )
        val_d = MyCCT_Dataset(
            '../datasets/cct/eccv_18_annotation_files/cis_val_annotations.json',
            transform=MyCCT_Dataset.get_val_bbox_transform()
        )
        cis_test_d = MyCCT_Dataset(
            '../datasets/cct/eccv_18_annotation_files/cis_test_annotations.json',
            transform=MyCCT_Dataset.get_val_bbox_transform()
        )
        trans_test_d = MyCCT_Dataset(
            '../datasets/cct/eccv_18_annotation_files/trans_test_annotations.json',
            transform=MyCCT_Dataset.get_val_bbox_transform()
        )
        return train_d, [val_d, cis_test_d, trans_test_d]

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--max_epochs", type=int, default=100)
        parser.add_argument("--batch", type=int, default=64,
                            help="Batch size.")
        parser.add_argument("--val_batch", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--batch_split", type=int, default=1,
                            help="Number of batches to compute gradient on before updating weights.")
        parser.add_argument("--base_lr", type=float, default=0.0045)
        # parser.add_argument("--val_data", type=float, default=2000 / 25227)
        # parser.add_argument("--test_data", type=float, default=5000 / 30227)
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
            monitor='val_aupr',
        )

        args = dict()
        args['max_epochs'] = self.hparams.max_epochs
        args['checkpoint_callback'] = checkpoint_callback

        last_ckpt = pjoin(self.hparams.logdir, self.hparams.name, 'last.ckpt')
        if pexists(last_ckpt):
            args['resume_from_checkpoint'] = last_ckpt
        return args
