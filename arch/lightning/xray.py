import torch
import torch.nn.functional as F
from os.path import join as pjoin
from os.path import exists as pexists
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint
from torchxrayvision.models import DenseNet
from sklearn.metrics import average_precision_score, roc_auc_score

from .base_imagenet import ImageNetLightningModel
from ..data.xrayvision_datasets import Kaggle_Dataset, NIH_Dataset, CheX_Dataset, MIMIC_Dataset
from ..data.imagenet_datasets import MySubset, MyConcatDataset
from ..saliency_utils import get_grad_y, get_grad_sum, \
    get_grad_logp_sum, get_grad_logp_y, get_deeplift


class XRayLightningModel(ImageNetLightningModel):
    def init_setup(self):
        # Densenet 121
        self.model = DenseNet(num_classes=2,
                              in_channels=1,
                              growth_rate=32,
                              block_config=(6, 12, 24, 16),
                              num_init_features=64)

    def on_fit_start(self):
        if self.current_epoch >= self.hparams.max_epochs:
            self.my_logger.info('Already finish training! Exit!')
            exit()

    def training_step(self, batch, batch_idx):
        x, y = batch

        is_dict = isinstance(x, dict)
        if is_dict:
            has_bbox = (x['xs'] != -1)
            mask = self._generate_mask(
                x['imgs'], x['xs'], x['ys'], x['ws'], x['hs'])
            if self.hparams.inpaint == 'none' or (~has_bbox).all():
                x = x['imgs']
            else:
                impute_x = self.inpaint_model(x['imgs'][has_bbox], mask[has_bbox])
                impute_y = (-y[has_bbox] - 1)

                orig_x_len = len(x['imgs'])
                x = torch.cat([x['imgs'], impute_x], dim=0)
                # label -1 as negative of class 0, -2 as negative of class 1 etc...
                y = torch.cat([y, impute_y], dim=0)

        if not is_dict or self.hparams.get('reg', 'none') == 'none' \
                or (is_dict and (~has_bbox).all()):
            logits = self(x)
            reg_loss = logits.new_tensor(0.)
        else:
            if self.hparams.inpaint == 'none' or (~has_bbox).all():
                x_orig, y_orig = x, y
            else:
                x_orig, x_cf, y_orig = x[:orig_x_len], x[orig_x_len:], y[:orig_x_len]

            saliency_fn = eval(f"get_{self.hparams.get('reg_grad', 'grad_y')}")
            the_grad, logits = saliency_fn(x_orig, y_orig, self.model,
                                           is_training=True)
            if torch.all(the_grad == 0.):
                reg_loss = logits.new_tensor(0.)
            elif self.hparams.reg == 'gs':
                assert is_dict and self.hparams.inpaint != 'none'
                dist = x_orig.detach()[has_bbox] - x_cf
                cos_sim = self.my_cosine_similarity(the_grad, dist)

                reg_loss = (1. - cos_sim).mean().mul_(self.hparams.reg_coeff)
            elif self.hparams.reg == 'bbox_o':
                reg_loss = ((the_grad[has_bbox] * mask[has_bbox]) ** 2).mean()\
                    .mul_(self.hparams.reg_coeff)
            elif self.hparams.reg == 'bbox_f1':
                norm = (the_grad[has_bbox] ** 2).sum(dim=1, keepdim=True)
                norm = (norm - norm.min()) / norm.max()
                gnd_truth = (1. - mask[has_bbox])

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

            if self.hparams.inpaint != 'none' and has_bbox.any():
                cf_logits = self(x_cf)
                logits = torch.cat([logits, cf_logits], dim=0)

        c = self.counterfact_cri(logits, y)

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

        tqdm_dict = {'train_loss': c, 'reg_loss': reg_loss,
                     'train_acc1': acc1,
                     'train_cf_acc1': cf_acc1}
        output = OrderedDict({
            # first epoch not optimizing reg loss
            # 'loss': c + reg_loss if self.current_epoch > 0 else c,
            # 'loss': c + reg_loss if 1e-2 * reg_loss < c < 100 * reg_loss
            #     else c,
            'loss': c + reg_loss,
            'train_loss': c,
            'reg_loss': reg_loss,
            'acc1': acc1,
            'cf_acc1': cf_acc1,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
        })
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if isinstance(x, dict):
            x = x['imgs']

        logit = self(x)
        prefix = ['val', 'test', 'nih', 'mimic', 'cheX'][dataloader_idx]

        output = OrderedDict({
            f'{prefix}_logit': logit,
            f'{prefix}_y': y,
        })
        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}

        def cal_metrics(output, prefix='val'):
            val_logit = torch.cat([o[f'{prefix}_logit'] for o in output])
            val_y = torch.cat([o[f'{prefix}_y'] for o in output])
            tqdm_dict[f'{prefix}_loss'] = F.cross_entropy(val_logit, val_y, reduction='mean')
            tqdm_dict[f'{prefix}_acc1'], = self.accuracy(val_logit, val_y, topk=(1,))
            try:
                tqdm_dict[f'{prefix}_auc'] = roc_auc_score(
                    val_y.cpu().numpy(), (val_logit[:, 1] - val_logit[:, 0]).cpu().numpy()) * 100
                tqdm_dict[f'{prefix}_aupr'] = average_precision_score(
                    val_y.cpu().numpy(), (val_logit[:, 1] - val_logit[:, 0]).cpu().numpy()) * 100
            except ValueError as e: # only 1 class is present. Happens in sanity check
                self.my_logger.warn('Only 1 class is present!\n' + str(e))
                tqdm_dict[f'{prefix}_auc'] = -1.
                tqdm_dict[f'{prefix}_aupr'] = -1.

        cal_metrics(outputs[0], 'val')
        cal_metrics(outputs[1], 'test')
        cal_metrics(outputs[2], 'nih')
        cal_metrics(outputs[3], 'mimic')
        cal_metrics(outputs[4], 'cheX')

        result = {
            'progress_bar': tqdm_dict, 'log': tqdm_dict,
            'val_loss': tqdm_dict["val_loss"],
        }
        return result

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.base_lr,
            weight_decay=1e-5, amsgrad=True)
        return optim

    def _make_train_val_dataset(self):
        def cut_into_train_val_test(dset, train_tx, test_tx):
            test_d, train_d = self.sub_dataset(dset, self.hparams.test_data)
            val_d, train_d = self.sub_dataset(train_d, self.hparams.val_data)

            test_d.transform = val_d.transform = test_tx
            train_d.transform = train_tx
            return train_d, val_d, test_d

        if self.hparams.inpaint == 'none' and self.hparams.reg == 'none':
            all_dataset = Kaggle_Dataset(
                f"../datasets/kaggle/stage_2_train_images_jpg/",
                include='all')
            train_d, val_d, test_d = cut_into_train_val_test(
                all_dataset,
                train_tx=Kaggle_Dataset.get_train_transform(self.hparams.test_run),
                test_tx=Kaggle_Dataset.get_val_transform(self.hparams.test_run)
            )
        else:
            all_dataset = Kaggle_Dataset(
                f"../datasets/kaggle/stage_2_train_images_jpg/",
                include='all_bbox')
            train_d, val_d, test_d = cut_into_train_val_test(
                all_dataset,
                train_tx=Kaggle_Dataset.get_train_bbox_transform(self.hparams.test_run),
                test_tx=Kaggle_Dataset.get_val_bbox_transform(self.hparams.test_run)
            )

        # Add other datasets
        nih_d = NIH_Dataset(transform=NIH_Dataset.get_val_transform(self.hparams.test_run))
        mimic_d = MIMIC_Dataset(transform=MIMIC_Dataset.get_val_transform(self.hparams.test_run))
        cheX_d = CheX_Dataset(transform=CheX_Dataset.get_val_transform(self.hparams.test_run))

        return train_d, [val_d, test_d, nih_d, mimic_d, cheX_d]

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--max_epochs", type=int, default=50)
        parser.add_argument("--batch", type=int, default=64,
                            help="Batch size.")
        parser.add_argument("--val_batch", type=int, default=256,
                            help="Batch size.")
        parser.add_argument("--batch_split", type=int, default=1,
                            help="Number of batches to compute gradient on before updating weights.")
        parser.add_argument("--base_lr", type=float, default=0.003)
        parser.add_argument("--val_data", type=float, default=2000 / 25227)
        parser.add_argument("--test_data", type=float, default=5000 / 30227)
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
            monitor='val_auc',
        )

        args = dict()
        args['max_epochs'] = self.hparams.max_epochs
        args['checkpoint_callback'] = checkpoint_callback

        last_ckpt = pjoin(self.hparams.logdir, self.hparams.name, 'last.ckpt')
        if pexists(last_ckpt):
            args['resume_from_checkpoint'] = last_ckpt
        return args
