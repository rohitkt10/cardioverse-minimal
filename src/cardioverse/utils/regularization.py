import torch
import torch.nn as nn


def lp_regularizer(model, p=2):
    """
    Compute Lp norm regularization.

    Excludes BatchNorm layers from regularization.
    """
    exclude = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    reg = 0.0
    for module in model.modules():
        if isinstance(module, exclude):
            continue
        if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
            reg += torch.norm(module.weight, p=p) ** p
    return reg
