import argparse, json, random, types
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
import configs.default as cfg
from data import CocoDetectionDataset, collate_fn, MixupCollate
from engine.optim import build_optimizer, build_scheduler, set_warmup_lr
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


def format_size(num_bytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024 or unit == "GiB":
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024


def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    print(
        "Model initialized: "
        f"{model.__class__.__name__} | parameters={total_params:,} | "
        f"trainable={trainable_params:,} | frozen={total_params - trainable_params:,} | "
        f"parameter+buffer size={format_size(parameter_bytes + buffer_bytes)}"
    )


def image_ids(targets):
    return [int(t['image_id'].item()) for t in targets]


def all_finite(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(all_finite(v) for v in value)
    return True


def save_checkpoint(path, model, optimizer, scheduler, epoch, batch_index, global_update, epoch_complete):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'batch_index': batch_index,
                'global_update': global_update, 'epoch_complete': epoch_complete,
                'config': {k: str(v) for k, v in vars(cfg).items() if not k.startswith('_')}}, path)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", default=cfg.resume); parser.add_argument("--device", default=cfg.device)
    args = parser.parse_args(); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data_cfg = snapshot_config(cfg)
    train_set = CocoDetectionDataset(cfg.train_images, cfg.train_annotations, data_cfg, True)
    val_set = CocoDetectionDataset(cfg.val_images, cfg.val_annotations, data_cfg, False)
    train_collate = MixupCollate(data_cfg)
    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True,
                              collate_fn=train_collate, pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_fn,
                            pin_memory=True, worker_init_fn=seed_worker)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"Data: {len(train_set):,} training samples / {len(val_set):,} validation samples | "
          f"{len(train_loader):,} training batches / {len(val_loader):,} validation batches")
    model = CustomDETR().to(device)
    print_model_summary(model)
    matcher = HungarianMatcher(cfg.matcher_class_cost, cfg.matcher_bbox_cost, cfg.matcher_giou_cost,
                               cfg.matcher_focal_alpha, cfg.matcher_focal_gamma)
    criterion = SetCriterion(cfg.num_class, matcher, {"loss_mal": cfg.loss_mal_weight, "loss_bbox": cfg.loss_bbox_weight, "loss_giou": cfg.loss_giou_weight}, cfg.mal_alpha, cfg.mal_gamma, cfg.aux_loss)
    optimizer, scheduler = build_optimizer(model, cfg), None; scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    start = 0; resume_batch = 0; global_update = 0
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False); model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        complete = checkpoint.get('epoch_complete', True)
        start = checkpoint['epoch'] + int(complete)
        resume_batch = 0 if complete else checkpoint.get('batch_index', 0)
        global_update = checkpoint.get('global_update', start * len(train_loader))
        print(f'Resumed from {args.resume} at epoch {start + 1}/{cfg.epochs}, batch {resume_batch + 1:,}')
    for epoch in range(start, cfg.epochs):
        train_set.set_epoch(epoch); train_collate.set_epoch(epoch); model.train(); running = 0.0
        batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="batch", dynamic_ncols=True) if tqdm else train_loader
        if tqdm is None:
            print(f"Epoch {epoch + 1}/{cfg.epochs}: training {len(train_loader):,} batches (install tqdm for a progress bar)")
        for batch_index, (images, targets) in enumerate(batches, start=1):
            if epoch == start and batch_index <= resume_batch: continue
            images, targets = images.to(device, non_blocking=True), move_targets(targets, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                outputs = model(images, targets)
            if not all_finite(outputs) or not all_finite(targets):
                print(f'WARNING: skipped non-finite batch {batch_index:,}; image_ids={image_ids(targets)}')
                continue
            try:
                losses = criterion(outputs, targets); loss = sum(losses.values())
            except FloatingPointError as exc:
                print(f'WARNING: skipped batch {batch_index:,}: {exc}; image_ids={image_ids(targets)}')
                continue
            if not torch.isfinite(loss):
                print(f'WARNING: skipped non-finite loss at batch {batch_index:,}; image_ids={image_ids(targets)}')
                continue
            if global_update <= cfg.warmup_updates:
                set_warmup_lr(optimizer, global_update, cfg.warmup_updates, cfg.lr, cfg.backbone_lr)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                print(f'WARNING: optimizer step skipped for non-finite gradients at batch {batch_index:,}')
            scaler.step(optimizer); scaler.update()
            if torch.isfinite(grad_norm): global_update += 1
            running += loss.item()
            if cfg.checkpoint_interval_updates > 0 and global_update > 0 and global_update % cfg.checkpoint_interval_updates == 0:
                save_checkpoint(out / 'last.pt', model, optimizer, scheduler, epoch, batch_index,
                                global_update, epoch_complete=False)
            if tqdm:
                main_lr = next(g["lr"] for g in optimizer.param_groups if not g["is_backbone"])
                backbone_lr = next(g["lr"] for g in optimizer.param_groups if g["is_backbone"])
                batches.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running / batch_index:.4f}",
                                    lr=f"{main_lr:.2e}", backbone_lr=f"{backbone_lr:.2e}")
            elif batch_index == 1 or batch_index == len(train_loader) or batch_index % max(len(train_loader) // 10, 1) == 0:
                print(f"  batch {batch_index:,}/{len(train_loader):,} | loss={loss.item():.4f} | "
                      f"avg_loss={running / batch_index:.4f}")
        print("Running validation...")
        val_loss = evaluate_loss(model, criterion, val_loader, device); scheduler.step(val_loss)
        coco_stats = coco_evaluate(model, val_loader, device) if cfg.run_coco_eval and (epoch + 1) % cfg.eval_every == 0 else None
        log = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1), "val_loss": val_loss,
               "lr": [g["lr"] for g in optimizer.param_groups]}
        if coco_stats is not None: log["coco_ap"] = coco_stats[:6]
        print("Epoch summary: " + json.dumps(log))
        save_checkpoint(out / 'last.pt', model, optimizer, scheduler, epoch, len(train_loader),
                        global_update, epoch_complete=True)
        print(f"Checkpoint saved to {out / 'last.pt'}")
        base_lrs = [g["lr"] for g in optimizer.param_groups if not g["is_backbone"]]
        if min(base_lrs) <= cfg.min_lr: print("minimum learning rate reached; stopping"); break


if __name__ == "__main__": main()
