import json, random
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from .transforms import final_transform, mosaic, mixup


class CocoDetectionDataset(Dataset):
    def __init__(self, images, annotations, cfg, train=True):
        self.root, self.cfg, self.train, self.epoch = Path(images), cfg, train, 0
        self.annotations_path = Path(annotations)
        data = json.loads(Path(annotations).read_text())
        self.images = data["images"]
        by_image = {x["id"]: [] for x in self.images}
        for ann in data["annotations"]:
            if not ann.get("iscrowd", 0): by_image.setdefault(ann["image_id"], []).append(ann)
        self.annotations = by_image
        cat_ids = sorted(c["id"] for c in data["categories"])
        self.category_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.label_to_category = {v: k for k, v in self.category_to_label.items()}

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return len(self.images)

    def load_item(self, index):
        info = self.images[index]
        image = Image.open(self.root / info["file_name"]).convert("RGB")
        anns = self.annotations.get(info["id"], [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h]); labels.append(self.category_to_label[ann["category_id"]])
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(labels, dtype=torch.long), "image_id": torch.tensor(info["id"]),
                  "orig_size": torch.tensor([info["height"], info["width"]])}
        return image, target

    def _composite(self, index, strong):
        sample = self.load_item(index)
        if strong and random.random() < self.cfg.mosaic_prob:
            samples = [sample] + [self.load_item(random.randrange(len(self))) for _ in range(3)]
            sample = mosaic(samples, self.cfg.mosaic_short_side, self.cfg.mosaic_scale, self.cfg.mosaic_translate)
        return sample

    def __getitem__(self, index):
        strong = self.train and self.epoch < self.cfg.strong_aug_stop_epoch
        sample = self._composite(index, strong)
        if strong and random.random() < self.cfg.mixup_prob:
            sample = mixup(sample, self._composite(random.randrange(len(self)), strong), self.cfg.mixup_alpha)
        image_id = self.images[index]["id"]
        image, target = final_transform(*sample, self.cfg.img_size, strong, self.cfg, augment=self.train)
        target["image_id"] = torch.tensor(image_id)
        target.setdefault("orig_size", torch.tensor([self.images[index]["height"], self.images[index]["width"]]))
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
