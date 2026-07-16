import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from data.box_ops import cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, class_cost=2.0, bbox_cost=5.0, giou_cost=2.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__(); self.class_cost, self.bbox_cost, self.giou_cost = class_cost, bbox_cost, giou_cost
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        results = []
        for logits, boxes, target in zip(outputs["pred_logits"], outputs["pred_boxes"], targets):
            labels, target_boxes = target["labels"], target["boxes"]
            if not len(labels):
                empty = torch.empty(0, dtype=torch.long, device=boxes.device); results.append((empty, empty)); continue
            prob = logits.sigmoid()
            neg_cost = (1 - self.focal_alpha) * prob.pow(self.focal_gamma) * -(1 - prob + 1e-8).log()
            pos_cost = self.focal_alpha * (1 - prob).pow(self.focal_gamma) * -(prob + 1e-8).log()
            class_cost = (pos_cost - neg_cost)[:, labels]
            bbox_cost = torch.cdist(boxes, target_boxes, p=1)
            giou_cost = -generalized_box_iou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(target_boxes))
            cost = self.class_cost * class_cost + self.bbox_cost * bbox_cost + self.giou_cost * giou_cost
            i, j = linear_sum_assignment(cost.detach().cpu())
            results.append((torch.as_tensor(i, device=boxes.device), torch.as_tensor(j, device=boxes.device)))
        return results
