import torch.nn as nn
import torch


class InpaintTemplate(nn.Module):
    def forward(self, x, mask):
        return self.impute_missing_imgs(x, mask)

    def impute_missing_imgs(self, x, mask):
        raise NotImplementedError('Need to implement this')

    def reset(self):
        pass


class MeanInpainter(InpaintTemplate):
    '''
    Just put 0 to impute. (As grey in Imagenet)
    '''
    def impute_missing_imgs(self, x, mask):
        return x * mask

    def generate_background(self, x, mask):
        return x.new(x.shape).zero_()


class LocalMeanInpainter(InpaintTemplate):
    def __init__(self, window=15):
        super(LocalMeanInpainter, self).__init__()
        self.window = window

        padding = int((window - 1) / 2)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=window, padding=padding, bias=False)
        self.conv1.weight.data.fill_(1.)

        self.num_of_existing_pixels = None

    def impute_missing_imgs(self, x, mask):
        mean_for_each_pixel = self.generate_background(x, mask)
        return x * mask + mean_for_each_pixel * (1. - mask)

    def generate_background(self, x, mask):
        sum_of_adjaent_pixels = self.conv1(x.view(-1, 1, *x.shape[2:])).data.view(*x.shape)

        if self.num_of_existing_pixels is None:
            self.num_of_existing_pixels = self.conv1(x.new_ones(x.shape[0], 1, *x.shape[2:])).data

        mean_for_each_pixel = sum_of_adjaent_pixels / self.num_of_existing_pixels
        return mean_for_each_pixel


class BlurryInpainter(InpaintTemplate):
    def __init__(self):
        super(BlurryInpainter, self).__init__()
        self.blurred_img = None

    def reset(self):
        self.blurred_img = None

    def impute_missing_imgs(self, x, mask):
        backgnd = self.generate_background(x, mask)
        return x * mask + backgnd * (1. - mask)

    def generate_background(self, x, mask):
        if self.blurred_img is None:
            self.blurred_img = self.blur_pytorch_img(x[0])
            self.blurred_img = self.blurred_img.unsqueeze(0)
        return self.blurred_img

    @staticmethod
    def blur_pytorch_img(pytorch_img):
        import cv2
        assert pytorch_img.ndimension() == 3

        np_img = pytorch_img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        background = cv2.GaussianBlur(np_img, (0, 0), 10)
        background = background[:, :, ::-1].tolist()

        py_back_img = pytorch_img.new(background).permute(2, 0, 1)
        return py_back_img


class RandomColorWithNoiseInpainter(InpaintTemplate):
    def __init__(self, color_mean=(0.485, 0.456, 0.406), color_std=(0.229, 0.224, 0.225)):
        super(RandomColorWithNoiseInpainter, self).__init__()
        self.color_mean = color_mean
        self.color_std = color_std

    def impute_missing_imgs(self, x, mask):
        background = self.generate_background(x, mask)
        return x * mask + background * (1. - mask)

    def generate_background(self, x, mask):
        random_img = x.new(x.size(0), 3, 1, 1).uniform_().repeat(1, 1, x.size(2), x.size(3))
        random_img += x.new(*x.size()).normal_(0, 0.2)
        random_img.clamp_(0., 1.)

        for c in [0, 1, 2]:
            random_img[:, c, :, :].sub_(self.color_mean[c]).div_(self.color_std[c])
        return random_img
