import torch


def cxcywh_to_xyxy(boxes):
    c, s = boxes[..., :2], boxes[..., 2:]
    return torch.cat((c - s / 2, c + s / 2), -1)


def xyxy_to_cxcywh(boxes):
    lo, hi = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((lo + hi) / 2, hi - lo), -1)


def box_area(boxes):
    return (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)


def box_iou(boxes1, boxes2):
    area1, area2 = box_area(boxes1), box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-7), union


def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    area = (rb - lt).clamp(min=0).prod(2)
    return iou - (area - union) / area.clamp(min=1e-7)
