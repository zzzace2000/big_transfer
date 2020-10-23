import os
from os.path import join as pjoin, exists as pexists
import pandas as pd
import numpy as np
import json

from arch.lightning.base_imagenet import ImageNetLightningModel
from arch.lightning.xray import XRayLightningModel
from arch.lightning.cct import CCTLightningModel
from arch.lightning.in9 import IN9LightningModel, IN9LLightningModel


for d in os.listdir('./models/'):
    result_f = pjoin('models', d, 'results.tsv')
    if not pexists(result_f):
        # raise Exception('WIERD!!!! No results.tsv found in the directory %s.' % d)
        print('WIERD!!!! No results.tsv found in the directory %s.' % d)
        continue

    if not pexists(pjoin('models', d, 'hparams.json')):
        print('No hparams exist. Have not finished!')
        continue

    is_finished_file = pjoin('models', d, 'SLURM_JOB_FINISHED')
    if not pexists(is_finished_file):
        print('Skip directory %s since job has not finished!' % d)
        continue

    model_cls = json.load(open(pjoin('models', d, 'hparams.json')))['pl_model']
    model_cls = eval(model_cls)
    if not model_cls.is_finished_run(pjoin('models', d)):
        print('Skip directory %s since job has not finished!' % d)
        continue

    bpath = pjoin('models', d, 'best.ckpt')

    def get_best_path(by='val_acc1'):
        df = pd.read_csv(result_f, delimiter='\t')
        best_idx = int(np.argmax(df[by].values))
        best_record = df.iloc[best_idx].to_dict()

        best_filename = 'epoch=%d.ckpt' % best_record['epoch']
        return best_filename

    def create_symlink(src, bpath):
        if os.path.islink(bpath): # exists already
            if os.path.realpath(bpath) == src:
                return
            os.unlink(bpath)
        elif pexists(bpath): # real file exists
            return
        os.symlink(src, bpath)

    if 'imageneta' in d: # use last chkpt as the best
        create_symlink('last.ckpt', bpath)
    elif 'in9' in d: # val_acc1
        best_filename = get_best_path(by='val_acc1')

        tmp = pjoin('models', d, best_filename)
        assert pexists(tmp), '%s not exists!' % tmp
        create_symlink(best_filename, bpath)
    elif 'cct' in d or 'xray' in d: # val_auc
        best_filename = get_best_path(by='val_auc')
        tmp = pjoin('models', d, best_filename)
        if not pexists(tmp):
            best_filename = get_best_path(by='val_acc1')
            tmp2 = pjoin('models', d, best_filename)

            assert pexists(tmp2), '%s and %s not exists!' % (tmp2, tmp)

        create_symlink(best_filename, bpath)
    else:
        raise NotImplementedError('UNKNOWN dir %s' % d)

    print('Finish %s' % bpath)
