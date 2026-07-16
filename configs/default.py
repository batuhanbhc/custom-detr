"""Default COCO training configuration.

All values are plain module attributes on purpose: edit this file or override them
with command-line flags in ``train.py``.
"""
from pathlib import Path

# Data
data_root = Path(__file__).resolve().parents[1] / "data" / "coco"
train_images = data_root / "train2017"
val_images = data_root / "val2017"
train_annotations = data_root / "annotations/instances_train2017.json"
val_annotations = data_root / "annotations/instances_val2017.json"
img_size = 640
batch_size = 4
num_workers = 4
num_class = 80

# Augmentation (DEIM-style, mosaic/mixup precede the single-image pipeline)
mosaic_prob = 0.5
mosaic_short_side = 320
mosaic_scale = (0.8, 1.2)
mosaic_translate = (0.1, 0.1)
mixup_prob = 0.5
mixup_alpha = 1.5
photometric_prob = 0.5
zoom_out_prob = 0.5
iou_crop_prob = 0.8
horizontal_flip_prob = 0.5
min_box_size = 1.0

# Two-phase schedule
epochs = 60
strong_aug_stop_epoch = 58

# Model
d_model = 256
num_heads = 8
num_levels = 3
num_enc_layers = 1
pre_norm = False
expansion = 4.0
ffn_dropout = 0.0
msa_dropout = 0.0
top_k = 300
base_anchor_width = 0.05
base_anchor_height = 0.05
num_decoder_layers = 6
k_list = [4, 6, 4]
decoder_expansion = 4.0
decoder_pre_norm = True
decoder_ffn_dropout = 0.0
decoder_msa_dropout = 0.0

# Contrastive denoising (one positive and one negative copy per GT/group)
num_denoising = 100
label_noise_ratio = 0.5
box_noise_scale = 1.0

# Hungarian matching and losses (DEIM / RT-DETR defaults)
matcher_class_cost = 2.0
matcher_bbox_cost = 5.0
matcher_giou_cost = 2.0
matcher_focal_alpha = 0.25
matcher_focal_gamma = 2.0
mal_alpha = 0.2
mal_gamma = 2.0
loss_mal_weight = 1.0
loss_bbox_weight = 5.0
loss_giou_weight = 2.0
aux_loss = True

# Optimization
lr = 2e-4
backbone_lr = 2e-5
weight_decay = 1e-4
betas = (0.9, 0.999)
grad_clip_norm = 0.1
amp = True
plateau_factor = 0.5
plateau_patience = 3
plateau_threshold = 1e-3
min_lr = 1e-7

# Runtime
device = "cuda"
seed = 0
output_dir = Path("outputs/default")
resume = None
eval_every = 1
run_coco_eval = True
