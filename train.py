import argparse, json, random, types
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import configs.default as cfg
from data import CocoDetectionDataset, collate_fn
from engine.optim import build_optimizer, build_scheduler
from engine.evaluate import evaluate_loss, coco_evaluate
from loss import HungarianMatcher, SetCriterion
from model import CustomDETR


def move_targets(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def seed_worker(worker_id):
    worker_seed = torch.utils.data.get_worker_info().seed % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def snapshot_config(cfg):
    """Plain, picklable copy of the config module's attributes.

    DataLoader workers (spawn/forkserver start methods, the default on macOS and
    increasingly on Linux) pickle the Dataset to hand it to worker processes; a
    live module object such as ``configs.default`` cannot be pickled.
    """
    return types.SimpleNamespace(**{k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", default=cfg.resume); parser.add_argument("--device", default=cfg.device)
    args = parser.parse_args(); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data_cfg = snapshot_config(cfg)
    train_set = CocoDetectionDataset(cfg.train_images, cfg.train_annotations, data_cfg, True)
    val_set = CocoDetectionDataset(cfg.val_images, cfg.val_annotations, data_cfg, False)
    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True,
                              collate_fn=collate_fn, pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_fn,
                            pin_memory=True, worker_init_fn=seed_worker)
    model = CustomDETR().to(device)
    matcher = HungarianMatcher(cfg.matcher_class_cost, cfg.matcher_bbox_cost, cfg.matcher_giou_cost,
                               cfg.matcher_focal_alpha, cfg.matcher_focal_gamma)
    criterion = SetCriterion(cfg.num_class, matcher, {"loss_mal": cfg.loss_mal_weight, "loss_bbox": cfg.loss_bbox_weight, "loss_giou": cfg.loss_giou_weight}, cfg.mal_alpha, cfg.mal_gamma, cfg.aux_loss)
    optimizer, scheduler = build_optimizer(model, cfg), None; scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    start = 0; out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"]); start = checkpoint["epoch"] + 1
    for epoch in range(start, cfg.epochs):
        train_set.set_epoch(epoch); model.train(); running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), move_targets(targets, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                losses = criterion(model(images, targets), targets); loss = sum(losses.values())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer); scaler.update(); running += loss.item()
        val_loss = evaluate_loss(model, criterion, val_loader, device); scheduler.step(val_loss)
        coco_stats = coco_evaluate(model, val_loader, device) if cfg.run_coco_eval and (epoch + 1) % cfg.eval_every == 0 else None
        log = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1), "val_loss": val_loss,
               "lr": [g["lr"] for g in optimizer.param_groups]}
        if coco_stats is not None: log["coco_ap"] = coco_stats[:6]
        print(json.dumps(log))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "config": {k: str(v) for k, v in vars(cfg).items() if not k.startswith("_")}}, out / "last.pt")
        base_lrs = [g["lr"] for g in optimizer.param_groups if not g["is_backbone"]]
        if min(base_lrs) <= cfg.min_lr: print("minimum learning rate reached; stopping"); break


if __name__ == "__main__": main()
