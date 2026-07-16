import torch
import torch.nn.functional as F


def matchability_aware_loss(logits, target_classes, target_ious, num_boxes, alpha=0.2, gamma=2.0):
    """DEIM MAL: IoU^gamma targets for positives, confidence-weighted negatives."""
    num_classes = logits.shape[-1]
    one_hot = F.one_hot(target_classes, num_classes + 1)[..., :-1].to(logits.dtype)
    scores = target_ious.unsqueeze(-1) * one_hot
    scores = scores.pow(gamma)
    pred = logits.sigmoid().detach()
    weight = alpha * pred.pow(gamma) * (1 - one_hot) + one_hot
    loss = F.binary_cross_entropy_with_logits(logits, scores, weight=weight, reduction="none")
    return loss.mean(1).sum() * logits.shape[1] / max(float(num_boxes), 1.0)
