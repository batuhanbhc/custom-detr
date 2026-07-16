import torch
from data.box_ops import cxcywh_to_xyxy


@torch.no_grad()
def evaluate_loss(model, criterion, loader, device):
    model.eval(); total = 0.0
    for images, targets in loader:
        images = images.to(device); targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        total += sum(criterion(model(images), targets).values()).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def coco_evaluate(model, loader, device):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval(); results = []
    for images, targets in loader:
        outputs = model(images.to(device)); scores = outputs["pred_logits"].sigmoid()
        confidence, labels = scores.max(-1)
        for b, target in enumerate(targets):
            h, w = target["orig_size"].tolist(); boxes = cxcywh_to_xyxy(outputs["pred_boxes"][b]).cpu()
            boxes *= torch.tensor([w, h, w, h]); boxes[:, 2:] -= boxes[:, :2]
            for box, score, label in zip(boxes, confidence[b].cpu(), labels[b].cpu()):
                results.append({"image_id": int(target["image_id"]), "category_id": loader.dataset.label_to_category[int(label)],
                                "bbox": box.tolist(), "score": float(score)})
    coco = COCO(str(loader.dataset.annotations_path)); pred = coco.loadRes(results)
    evaluator = COCOeval(coco, pred, "bbox"); evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return evaluator.stats.tolist()
