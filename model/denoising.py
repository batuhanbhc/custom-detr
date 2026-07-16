import torch
from data.box_ops import cxcywh_to_xyxy, xyxy_to_cxcywh


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(eps, 1 - eps); return torch.log(x / (1 - x))


def build_cdn(targets, num_classes, num_queries, embedding, num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0):
    counts = [len(t["labels"]) for t in targets]; max_gt = max(counts, default=0)
    if not max_gt: return None, None, None, None
    device = targets[0]["labels"].device; groups = max(1, num_denoising // max_gt); bs = len(targets)
    labels = torch.full((bs, max_gt), num_classes, dtype=torch.long, device=device)
    boxes = torch.zeros(bs, max_gt, 4, device=device); valid = torch.zeros(bs, max_gt, dtype=torch.bool, device=device)
    for b, target in enumerate(targets):
        n = counts[b]; labels[b, :n], boxes[b, :n], valid[b, :n] = target["labels"], target["boxes"], True
    labels = labels.tile(1, 2 * groups); boxes = boxes.tile(1, 2 * groups, 1); valid = valid.tile(1, 2 * groups)
    negative = torch.zeros(bs, max_gt * 2, 1, device=device); negative[:, max_gt:] = 1; negative = negative.tile(1, groups, 1)
    positive = ((1 - negative.squeeze(-1)).bool() & valid)
    positive_idx = torch.split(torch.nonzero(positive)[:, 1], [n * groups for n in counts])
    if label_noise_ratio:
        mask = (torch.rand_like(labels, dtype=torch.float) < label_noise_ratio * .5) & valid
        labels = torch.where(mask, torch.randint(0, num_classes, labels.shape, device=device), labels)
    if box_noise_scale:
        xyxy = cxcywh_to_xyxy(boxes); diff = boxes[..., 2:].tile(1, 1, 2) * .5 * box_noise_scale
        sign = torch.randint_like(boxes, 0, 2) * 2 - 1
        magnitude = torch.rand_like(boxes) + negative
        boxes = xyxy_to_cxcywh((xyxy + sign * magnitude * diff).clamp(0, 1))
    total = max_gt * 2 * groups; mask = torch.zeros(total + num_queries, total + num_queries, dtype=torch.bool, device=device)
    mask[total:, :total] = True
    for g in range(groups):
        lo, hi = 2 * max_gt * g, 2 * max_gt * (g + 1); mask[lo:hi, :lo] = True; mask[lo:hi, hi:total] = True
    meta = {"dn_positive_idx": positive_idx, "dn_num_group": groups, "dn_num_split": [total, num_queries]}
    return embedding(labels), inverse_sigmoid(boxes), mask, meta
