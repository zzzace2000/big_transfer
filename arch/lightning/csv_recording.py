import pytorch_lightning as pl
from os.path import join as pjoin
from os.path import exists as pexists
from collections import OrderedDict
import pandas as pd
import numpy as np
from argparse import Namespace

from ..utils import output_csv


class CSVRecordingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        csv_dict = OrderedDict()
        csv_dict['epoch'] = pl_module.current_epoch
        csv_dict['global_step'] = pl_module.global_step
        # In the metrics csv, the epoch is lagged by 1. Remove it.
        csv_dict.update({k: v for k, v in metrics.items()
                         if k not in ['epoch', 'global_step']})

        result_f = pjoin(pl_module.hparams.logdir, pl_module.hparams.name,
                         'results.tsv')
        output_csv(result_f, csv_dict, delimiter='\t')

    def on_train_end(self, trainer, pl_module):
        result_f = pjoin(pl_module.hparams.logdir, pl_module.hparams.name,
                         'results.tsv')
        if not pexists(result_f):
            return

        df = pd.read_csv(result_f, delimiter='\t')

        func = {'min': np.argmin, 'max': np.argmax}[trainer.checkpoint_callback.mode]
        best_idx = int(func(df[trainer.checkpoint_callback.monitor].values))

        best_metric = df.iloc[best_idx].to_dict()

        csv_dict = OrderedDict()
        csv_dict['name'] = pl_module.hparams.name
        csv_dict.update(best_metric)
        csv_dict.update(
            vars(pl_module.hparams) if isinstance(pl_module.hparams, Namespace)
            else pl_module.hparams)

        postfix = '_test' if pl_module.hparams.test_run else ''
        fname = pjoin(pl_module.hparams.result_dir,
                      f'{pl_module.__class__.__name__}_results2{postfix}.tsv')
        output_csv(fname,
                   csv_dict, delimiter='\t')
