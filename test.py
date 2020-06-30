from arch.Inpainting.CAInpainter import CAInpainter
from data_utils import ImagenetBoundingBoxFolder, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
import torchvision as tv
import torch
from os.path import join as pjoin
from data_utils import bbox_collate

# val_tx = tv.transforms.Compose([
#       Resize((256, 256)),
#       ToTensor(),
#       Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
ca = CAInpainter(batch_size=24, checkpoint_dir='./inpainting_models/release_imagenet_256/')


val_tx = tv.transforms.Compose([
  Resize((32, 32)),
  RandomCrop((28, 28)),
  RandomHorizontalFlip(),
  ToTensor(),
  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

data_dir = '../datasets/imagenet/'
valid_set = ImagenetBoundingBoxFolder(
    root=pjoin(data_dir, 'train_objectnet/'),
    bbox_file=pjoin(data_dir, 'LOC_train_solution.csv'),
    transform=val_tx)


valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=128, shuffle=True,
      num_workers=1, pin_memory=True, drop_last=False,
      collate_fn=bbox_collate)

while True:
    for samples, targets in valid_loader:
        print('hahaha')

print('asd')
print('asd')