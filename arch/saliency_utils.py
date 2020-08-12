import torch
from .myhack import HackGradAndOutputs, MyDeepLift
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)


def _get_grad_base(x, y, model, callback, is_training=True, **kwargs):
    with torch.enable_grad():
        x.requires_grad_(True)
        logits = model(x)

        target = callback(logits, y)
        the_grad = torch.autograd.grad(
            target, x,
            retain_graph=is_training,
            create_graph=is_training)[0]
        x.requires_grad_(False)
    return the_grad, logits


def get_grad_y(x, y, model, is_training=True, **kwargs):
    ''' Gradient supervision (2020) '''
    def callback(logits, y):
        return logits[:, y].sum()

    return _get_grad_base(x, y, model, callback=callback,
                          is_training=is_training, **kwargs)


def get_grad_sum(x, y, model, is_training=True, **kwargs):
    def callback(logits, y):
        return logits.sum()

    return _get_grad_base(x, y, model, callback=callback,
                          is_training=is_training, **kwargs)


def get_grad_logp_y(x, y, model, is_training=True, **kwargs):
    def callback(logits, y):
        logp = torch.log_softmax(logits, dim=1)
        return logp[:, y].sum()

    return _get_grad_base(x, y, model, callback=callback,
                          is_training=is_training, **kwargs)


def get_grad_logp_sum(x, y, model, is_training=True, **kwargs):
    ''' Learning right for the right reason (2017) '''
    def callback(logits, y):
        logp = torch.log_softmax(logits, dim=1)
        return logp.sum()

    return _get_grad_base(x, y, model, callback=callback,
                          is_training=is_training, **kwargs)


def get_deeplift(x, y, model, is_training=True, baselines=0.):
    with torch.enable_grad(), \
         HackGradAndOutputs(is_training=is_training) as hack:
        dl = MyDeepLift(model, kept_backward=True)
        attributions = dl.attribute(x, baselines,
                                    target=y, return_convergence_delta=False)
        dl.kept_backward = False
        bs = x.shape[0]
        return attributions, hack.output[:bs]
