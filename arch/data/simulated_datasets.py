import torch


class GaussianNoiseDataset(object):
    '''
    Generate a Gaussian noise with mean 0.5 and stdev 1,
    clip between 0 and 1, and normalize to -1 and 1.
    '''
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def make_loader(self, batch_size, shuffle=None, workers=None):
        num_times = (self.num_samples // batch_size)
        for _ in range(num_times):
            gn = torch.randn(batch_size, 3, 224, 224).add_(0.5)
            gn.clamp_(min=0., max=1.).sub_(0.5).mul_(2)
            yield gn, None

        residual = self.num_samples - num_times * batch_size
        if residual > 0:
            gn = torch.randn(residual, 3, 224, 224).add_(0.5)
            gn.clamp_(min=0., max=1.).sub_(0.5).mul_(2)
            yield gn, None

    def __len__(self):
        return self.num_samples


class UniformNoiseDataset(object):
    '''
    Generate a Gaussian noise with mean 0.5 and stdev 1,
    clip between 0 and 1, and normalize to -1 and 1.
    '''
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def make_loader(self, batch_size, shuffle, workers):
        num_times = (self.num_samples // batch_size)
        for _ in range(num_times):
            un = torch.rand(batch_size, 3, 224, 224).sub_(0.5).mul_(2)
            yield un, None

        residual = self.num_samples - num_times * batch_size
        if residual > 0:
            un = torch.rand(batch_size, 3, 224, 224).sub_(0.5).mul_(2)
            yield un, None

    def __len__(self):
        return self.num_samples
