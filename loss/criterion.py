import torch
from torch import nn
import torch.nn.functional as F
from data.box_ops import box_iou, cxcywh_to_xyxy, generalized_box_iou
from .mal import matchability_aware_loss


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weights, mal_alpha=0.2, mal_gamma=2.0, aux_loss=True):
        super().__init__(); self.num_classes, self.matcher, self.weights = num_classes, matcher, weights
        self.mal_alpha, self.mal_gamma, self.aux_loss = mal_alpha, mal_gamma, aux_loss

    def _loss(self, outputs, targets, indices, num_boxes, suffix=""):
        device = outputs["pred_logits"].device
        target_classes = torch.full(outputs["pred_logits"].shape[:2], self.num_classes, dtype=torch.long, device=device)
        target_ious = torch.zeros(outputs["pred_logits"].shape[:2], device=device)
        src, dst = [], []
        for b, (i, j) in enumerate(indices):
            if len(i):
                target_classes[b, i] = targets[b]["labels"][j]
                iou = box_iou(cxcywh_to_xyxy(outputs["pred_boxes"][b, i]), cxcywh_to_xyxy(targets[b]["boxes"][j]))[0].diag()
                target_ious[b, i] = iou.detach(); src.append(outputs["pred_boxes"][b, i]); dst.append(targets[b]["boxes"][j])
        mal = matchability_aware_loss(outputs["pred_logits"], target_classes, target_ious, num_boxes, self.mal_alpha, self.mal_gamma)
        if src:
            src, dst = torch.cat(src), torch.cat(dst)
            l1 = F.l1_loss(src, dst, reduction="sum") / num_boxes
            giou = (1 - generalized_box_iou(cxcywh_to_xyxy(src), cxcywh_to_xyxy(dst)).diag()).sum() / num_boxes
        else: l1 = outputs["pred_boxes"].sum() * 0; giou = l1
        return {"loss_mal" + suffix: mal, "loss_bbox" + suffix: l1, "loss_giou" + suffix: giou}

    def forward(self, outputs, targets):
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        losses = {}; main = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}
        losses.update(self._loss(main, targets, self.matcher(main, targets), num_boxes))
        # Every decoder layer and encoder top-k receive independent matching.
        if self.aux_loss:
            for layer in range(outputs["decoder_class_logits"].shape[1] - 1):
                out = {"pred_logits": outputs["decoder_class_logits"][:, layer], "pred_boxes": outputs["decoder_boxes"][:, layer]}
                losses.update(self._loss(out, targets, self.matcher(out, targets), num_boxes, f"_aux_{layer}"))
            enc = {"pred_logits": outputs["enc_topk_class_logits"], "pred_boxes": outputs["enc_topk_boxes"]}
            losses.update(self._loss(enc, targets, self.matcher(enc, targets), num_boxes, "_enc"))
        if outputs.get("dn_meta") is not None:
            meta = outputs["dn_meta"]
            for layer in range(outputs["dn_class_logits"].shape[1]):
                out = {"pred_logits": outputs["dn_class_logits"][:, layer], "pred_boxes": outputs["dn_boxes"][:, layer]}
                indices = []
                for b, positive in enumerate(meta["dn_positive_idx"]):
                    gt = torch.arange(len(targets[b]["labels"]), device=positive.device).repeat(meta["dn_num_group"])
                    indices.append((positive, gt))
                losses.update(self._loss(out, targets, indices, num_boxes * meta["dn_num_group"], f"_dn_{layer}"))
        weighted = {}
        for name, value in losses.items():
            base = "loss_mal" if name.startswith("loss_mal") else "loss_bbox" if name.startswith("loss_bbox") else "loss_giou"
            weighted[name] = value * self.weights[base]
        return weighted
