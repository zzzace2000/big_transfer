import os
from os.path import join as pjoin, exists as pexists
import pandas as pd
import json
import numpy as np
from collections import OrderedDict
from arch.utils import output_csv


all_names = [
    d for d in os.listdir('./models/')
    if d.startswith('1013_cct_random_frandom_bbox_o_reg100_')
    #    or d.startswith('0926_imageneta')
    #    or d.startswith('0922_2_imageneta')
    #    or d.startswith('0929')
]

for name in all_names:
    the_dir = 'models/%s' % name

    hparams_path = pjoin(the_dir, 'hparams.json')
    if not pexists(hparams_path):
        continue

    hparams = json.load(open(hparams_path))
    df = pd.read_csv(pjoin(the_dir, 'results.tsv'), delimiter='\t')

    best_idx = int(np.argmax(df['val_auc'].values))
    best_metric = df.iloc[best_idx].to_dict()

    # best_metric = df.iloc[-1].to_dict()

    csv_dict = OrderedDict()
    csv_dict['name'] = hparams['name']
    csv_dict['epoch'] = best_metric['epoch']
    if 'data_ratio' not in hparams:
        hparams['data_ratio'] = 1
    csv_dict.update(best_metric)
    csv_dict.update(hparams)

    fname = pjoin('results', f'CCTLightningModel_results.tsv')
    output_csv(fname, csv_dict, delimiter='\t')
