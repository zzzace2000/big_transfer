'''
Cleaned version for doing imputation by a pre-trained VAE.
'''
import torch
import torch.nn as nn
from torch.nn import functional as F


class VAEInpainter(torch.nn.Module):
    def __init__(self, in_mean=None, in_std=None):
        super(VAEInpainter, self).__init__()

        self.in_mean = in_mean
        self.in_std = in_std

        # Training mean
        self.training_mean = [0.485, 0.456, 0.406]
        self.training_std = [0.229, 0.224, 0.225]

        self.in_transform = torch.nn.Upsample(size=(224, 224), mode='bilinear')

        self.encode = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Start dilation
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # End Dilation
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1),
        )

        self.decode = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Upconvolution
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # Do normal convolution
            nn.Conv2d(32, 6, kernel_size=3, stride=1, padding=1, dilation=1),
        )

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    @staticmethod
    def decouple(params, channel_num=None):
        if channel_num is None:
            channel_num = int(params.size(1) / 2)

        mu = params[:, :channel_num, :, :]
        logvar = params[:, channel_num:, :, :]
        return mu, logvar

    def forward(self, x, mask):
        if self.in_mean is not None:
            x = x.clone()
            for i, (m, s, tm, ts) in enumerate(zip(
                    self.in_mean, self.in_std, self.training_mean, self.training_std)):
                x[:, i, :, :].mul_(s).add_(m - tm).div_(ts)

        orig_size = x.shape[2:]
        x = torch.cat((x.mul_(mask), 1. - mask), dim=1)

        if orig_size[0] != 224 or orig_size[1] != 224:
            x = self.in_transform(x)
            x[:, 3, :, :].round_()

        z_params = self.encode(x)
        mu, logvar = self.decouple(z_params)

        z = self.reparametrize(mu, logvar)

        x_params = self.decode(z)
        recon_x, _ = self.decouple(x_params)

        # clipping the maximum value to be in the range
        for c, (tm, ts) in enumerate(zip(self.training_mean, self.training_std)):
            recon_x[:, c, :, :].clamp_(min=(0. - tm) / ts, max=(1. - tm) / ts)

        if self.in_mean is not None:
            for i, (m, s, tm, ts) in enumerate(zip(
                    self.in_mean, self.in_std, self.training_mean, self.training_std)):
                recon_x[:, i, :, :].mul_(ts).add_(tm-m).div_(s)

        # clipping
        if orig_size[0] != 224 or orig_size[1] != 224:
            recon_x = F.interpolate(recon_x, size=orig_size, mode='bilinear').data

        return recon_x
