import torch.nn as nn

from captum.attr import (
    GradientShap,
    DeepLift,
    NoiseTunnel,
)

from arch.data_utils import ImagenetBoundingBoxFolder, Resize, ToTensor, Normalize
import torch
from os.path import join as pjoin
from torch.utils.data._utils.collate import default_collate
from arch.data_utils import bbox_collate
import pandas as pd
from torchvision.datasets import ImageFolder
import torchvision as tv
from arch.myhack import HackGradAndOutputs
import arch.models as models


# val_tx = tv.transforms.Compose([
#       Resize((256, 256)),
#       ToTensor(),
# #       Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
val_tx = tv.transforms.Compose([
  Resize((224, 224)),
  ToTensor(),
  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

data_dir = '../datasets/imagenet/'
valid_set = ImagenetBoundingBoxFolder(
      root=pjoin(data_dir, "val_objectnet"),
      bbox_file=pjoin(data_dir, 'LOC_val_solution.csv'),
      transform=val_tx)

valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=4, shuffle=True,
      num_workers=0, pin_memory=True, drop_last=False,
      collate_fn=bbox_collate)

samples, targets = next(iter(valid_loader))

model = models.KNOWN_MODELS['BiT-S-R50x1'](
    head_size=len(valid_set.classes), zero_head=False)

imgs = samples['imgs'].cuda()
model.cuda()

with HackGradAndOutputs() as hack:
    dl = DeepLift(model)
    attributions = dl.attribute(imgs, torch.zeros_like(imgs),
                                target=0, return_convergence_delta=False)
    print(hack)
    print(hack.output.shape)

