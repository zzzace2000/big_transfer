from captum.attr import (
    GradientShap,
    DeepLift,
    NoiseTunnel,
    GuidedGradCam,
)

import torch
from os.path import join as pjoin
import torchvision as tv
from arch.myhack import HackGradAndOutputs
from arch.data.cct_datasets import MyCCT_Dataset
from arch import models
from arch.inpainting.Baseline import BlurryInpainter, TileInpainter


# val_tx = tv.transforms.Compose([
#       Resize((256, 256)),
#       ToTensor(),
# #       Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

val_d = MyCCT_Dataset(
    '../datasets/cct/eccv_18_annotation_files/train_annotations.json',
    transform=MyCCT_Dataset.get_val_bbox_transform()
)

samples, target = val_d[0]
samples['imgs'] = (samples['imgs'] / 2) + 0.5

img = samples['imgs']
mask = torch.ones_like(img)
mask[:, samples['ys']:(samples['ys'] + samples['hs']),
    samples['xs']:(samples['xs'] + samples['ws'])] = 0.

inpainter = TileInpainter()
new_img = inpainter.impute_missing_imgs(
    img.unsqueeze_(0), mask.unsqueeze_(0))


model = models.KNOWN_MODELS['BiT-S-R50x1'](
    head_size=16, zero_head=False)
# model = tv.models.wide_resnet50_2(pretrained=True)

imgs = samples['imgs'].cuda()
model.cuda()

with HackGradAndOutputs() as hack:
    # dl = DeepLift(model)
    dl = GuidedGradCam(model)
    attributions = dl.attribute(imgs, torch.zeros_like(imgs),
                                target=0, return_convergence_delta=False)
    print(hack)
    print(hack.output.shape)
