import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
from .box_ops import xyxy_to_cxcywh


def resize_short(image, target, short=320):
    w, h = image.size
    scale = short / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    image = image.resize((nw, nh), Image.Resampling.BILINEAR)
    target = clone_target(target)
    target["boxes"] *= torch.tensor([scale, scale, scale, scale])
    return image, target


def clone_target(target):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()}


def sanitize(target, w, h, min_size=1.0):
    boxes = target["boxes"]
    boxes[:, 0::2].clamp_(0, w)
    boxes[:, 1::2].clamp_(0, h)
    keep = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & ((boxes[:, 3] - boxes[:, 1]) >= min_size)
    for key in ("boxes", "labels", "area", "iscrowd"):
        if key in target:
            target[key] = target[key][keep]
    return target


def mosaic(samples, short=320, scale_range=(0.8, 1.2), translate=(0.1, 0.1)):
    resized = [resize_short(*sample, short) for sample in samples]
    max_w = max(im.width for im, _ in resized)
    max_h = max(im.height for im, _ in resized)
    canvas = Image.new("RGB", (2 * max_w, 2 * max_h), (114, 114, 114))
    offsets = ((0, 0), (max_w, 0), (0, max_h), (max_w, max_h))
    boxes, labels = [], []
    for (im, target), (x, y) in zip(resized, offsets):
        canvas.paste(im, (x, y))
        boxes.append(target["boxes"] + torch.tensor([x, y, x, y]))
        labels.append(target["labels"])
    target = {"boxes": torch.cat(boxes), "labels": torch.cat(labels)}
    # DEIM's final RandomAffine: scale about center, then translate the mosaic.
    s = random.uniform(*scale_range)
    sw, sh = max(1, round(canvas.width * s)), max(1, round(canvas.height * s))
    scaled = canvas.resize((sw, sh), Image.Resampling.BILINEAR)
    target["boxes"] *= s
    out = Image.new("RGB", canvas.size, (114, 114, 114))
    dx = round((canvas.width - sw) / 2 + random.uniform(-translate[0], translate[0]) * canvas.width)
    dy = round((canvas.height - sh) / 2 + random.uniform(-translate[1], translate[1]) * canvas.height)
    out.paste(scaled, (dx, dy))
    target["boxes"] += torch.tensor([dx, dy, dx, dy])
    return out, sanitize(target, *out.size)


def mixup(a, b, alpha=1.5):
    image1, target1 = a
    image2, target2 = b
    if image2.size != image1.size:
        sx, sy = image1.width / image2.width, image1.height / image2.height
        image2 = image2.resize(image1.size, Image.Resampling.BILINEAR)
        target2 = clone_target(target2)
        target2["boxes"] *= torch.tensor([sx, sy, sx, sy])
    lam = random.betavariate(alpha, alpha)
    image = Image.blend(image1, image2, 1 - lam)
    return image, {"boxes": torch.cat((target1["boxes"], target2["boxes"])),
                   "labels": torch.cat((target1["labels"], target2["labels"]))}


def hue_jitter(image, factor):
    h, s, v = image.convert("HSV").split()
    h_arr = (np.array(h, dtype=np.int16) + round(factor * 255)) % 256
    h = Image.fromarray(h_arr.astype(np.uint8), mode="L")
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def channel_permute(image):
    bands = image.split()
    order = list(range(len(bands)))
    random.shuffle(order)
    return Image.merge(image.mode, [bands[i] for i in order])


def photometric(image, p=0.5):
    # Mirrors torchvision's RandomPhotometricDistort (each op independently gated at p).
    brightness, contrast, saturation, hue, permute = (random.random() < p for _ in range(5))
    contrast_before = random.random() < 0.5
    if brightness:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.875, 1.125))
    if contrast and contrast_before:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    if saturation:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))
    if hue:
        image = hue_jitter(image, random.uniform(-0.05, 0.05))
    if contrast and not contrast_before:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    if permute:
        image = channel_permute(image)
    return image


def zoom_out(image, target):
    scale = random.uniform(1.0, 4.0)
    nw, nh = round(image.width * scale), round(image.height * scale)
    left, top = random.randint(0, nw - image.width), random.randint(0, nh - image.height)
    out = Image.new("RGB", (nw, nh), (0, 0, 0)); out.paste(image, (left, top))
    target = clone_target(target); target["boxes"] += torch.tensor([left, top, left, top])
    return out, target


def random_iou_crop(image, target, trials=40, filter_mode="min_visibility", min_visibility=0.1):
    boxes = target["boxes"]
    if not len(boxes): return image, target
    options = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None)
    threshold = random.choice(options)
    if threshold is None: return image, target
    area = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    for _ in range(trials):
        w = random.uniform(0.3, 1.0) * image.width; h = random.uniform(0.3, 1.0) * image.height
        if not 0.5 <= w / h <= 2.0: continue
        left = random.uniform(0, image.width - w); top = random.uniform(0, image.height - h)
        crop = torch.tensor([left, top, left + w, top + h])
        inter_lt = torch.maximum(boxes[:, :2], crop[:2]); inter_rb = torch.minimum(boxes[:, 2:], crop[2:])
        inter = (inter_rb - inter_lt).clamp(min=0).prod(1)
        if filter_mode == "center":
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            keep = ((centers > crop[:2]) & (centers < crop[2:])).all(1)
        else:
            keep = (inter / area.clamp(min=1e-7)) > min_visibility
        if not keep.any(): continue
        crop_area = w * h
        if (inter[keep] / (area[keep] + crop_area - inter[keep])).max() < threshold: continue
        target = clone_target(target)
        target["boxes"], target["labels"] = boxes[keep] - torch.tensor([left, top, left, top]), target["labels"][keep]
        return image.crop((round(left), round(top), round(left + w), round(top + h))), sanitize(target, round(w), round(h))
    return image, target


def final_transform(image, target, size=640, strong=True, cfg=None, augment=True):
    if strong and random.random() < cfg.photometric_prob: image = photometric(image)
    if strong and random.random() < cfg.zoom_out_prob: image, target = zoom_out(image, target)
    if strong and random.random() < cfg.iou_crop_prob:
        image, target = random_iou_crop(
            image, target,
            filter_mode=cfg.iou_crop_filter,
            min_visibility=cfg.iou_crop_min_visibility,
        )
    if augment and random.random() < cfg.horizontal_flip_prob:
        image = F.hflip(image); boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]; target["boxes"] = boxes
    ow, oh = image.size
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    target["boxes"] *= torch.tensor([size / ow, size / oh, size / ow, size / oh])
    target = sanitize(target, size, size, cfg.min_box_size)
    target["boxes"] = xyxy_to_cxcywh(target["boxes"]) / size
    return F.to_tensor(image), target
