import torch
from torch import nn


def build_optimizer(model, cfg):
    groups = {}
    norm_params = {id(p) for m in model.modules() if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)) or m.__class__.__name__ == "LayerNorm" for p in m.parameters(recurse=False)}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        backbone = name.startswith("backbone.")
        no_decay = name.endswith(".bias") or id(param) in norm_params
        key = (backbone, no_decay); groups.setdefault(key, []).append(param)
    param_groups = [{"params": params, "lr": cfg.backbone_lr if backbone else cfg.lr,
                     "weight_decay": 0.0 if no_decay else cfg.weight_decay, "is_backbone": backbone}
                    for (backbone, no_decay), params in groups.items()]
    return torch.optim.AdamW(param_groups, betas=cfg.betas)


def set_warmup_lr(optimizer, update, warmup_updates, lr, backbone_lr):
    '''Linearly scale learning rates from zero over optimizer updates.'''
    scale = 1.0 if warmup_updates <= 0 else min(update / warmup_updates, 1.0)
    for group in optimizer.param_groups:
        group['lr'] = (backbone_lr if group['is_backbone'] else lr) * scale


def build_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience,
        threshold=cfg.plateau_threshold, min_lr=cfg.min_lr)
