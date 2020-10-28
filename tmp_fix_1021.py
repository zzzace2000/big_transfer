import os
from os.path import join as pjoin, exists as pexists
import pandas as pd
import numpy as np


for d in os.listdir('./models/'):
    bpath = pjoin('./models', d, 'best.ckpt')
    if d.startswith('1021'):
        print(d)
        if pexists(bpath):
            assert os.path.islink(bpath), bpath
            os.unlink(bpath)
            print('Remove %s' % bpath)
