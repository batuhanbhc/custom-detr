import torch
import torch.nn as nn

from .backbone import ResNetBackbone
from .encoder import HybridEncoder
from .query_selection import UncertaintyMinimalQuery
from .decoder import Decoder
import configs.default as cfg


class CustomDETR(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ResNetBackbone()

        self.encoder = HybridEncoder(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_enc_layers=cfg.num_enc_layers,
            pre_norm=cfg.pre_norm,
            expansion=cfg.expansion,
            ffn_dropout=cfg.ffn_dropout,
            msa_dropout=cfg.msa_dropout,
        )

        self.query_selector = UncertaintyMinimalQuery(
            d_model=cfg.d_model,
            num_class=cfg.num_class,
            k=cfg.top_k,
            base_h=cfg.base_anchor_height,
            base_w=cfg.base_anchor_width,
        )

        self.decoder = Decoder(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_levels=cfg.num_levels,
            k_list=cfg.k_list,
            num_classes=cfg.num_class,
            num_layers=cfg.num_decoder_layers,
            expansion=cfg.decoder_expansion,
            pre_norm=cfg.decoder_pre_norm,
            ffn_dropout=cfg.decoder_ffn_dropout,
            msa_dropout=cfg.decoder_msa_dropout,
        )

    @staticmethod
    def build_spatial_metadata(features):
        """
        features:
            tuple/list of tensors shaped [B, C, H_i, W_i]

        Returns:
            spatial_sizes:
                [num_levels, 2], storing [H_i, W_i]

            level_start_index:
                [num_levels], storing the starting token index of each level
                in the concatenated flattened memory.
        """

        device = features[0].device

        spatial_sizes = torch.tensor(
            [
                [feature.shape[-2], feature.shape[-1]]
                for feature in features
            ],
            dtype=torch.long,
            device=device,
        )

        tokens_per_level = (
            spatial_sizes[:, 0] * spatial_sizes[:, 1]
        )

        level_start_index = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=torch.long,
                    device=device,
                ),
                tokens_per_level.cumsum(dim=0)[:-1],
            ],
            dim=0,
        )

        return spatial_sizes, level_start_index

    def forward(self, x: torch.Tensor):
        backbone_features = self.backbone(x)

        encoder_features = self.encoder(backbone_features)
        # p3, p4, p5:
        # [B, D, H/8,  W/8]
        # [B, D, H/16, W/16]
        # [B, D, H/32, W/32]

        spatial_sizes, level_start_index = (
            self.build_spatial_metadata(encoder_features)
        )

        query_output = self.query_selector(
            encoder_features
        )

        decoder_output = self.decoder(
            query_tokens=query_output["query_tokens"],
            memory=query_output["memory"],
            reference_box_logits=query_output[
                "reference_box_logits"
            ],
            spatial_sizes=spatial_sizes,
            level_start_index=level_start_index,
            attn_mask=None,
        )

        return {
            # Final decoder output
            "pred_logits": decoder_output["pred_logits"],
            "pred_boxes": decoder_output["pred_boxes"],

            # All decoder layers, including final layer
            "decoder_class_logits": decoder_output[
                "all_class_logits"
            ],
            "decoder_boxes": decoder_output[
                "all_pred_boxes"
            ],

            # Encoder top-k auxiliary predictions
            "enc_topk_class_logits": query_output[
                "enc_topk_class_logits"
            ],
            "enc_topk_boxes": query_output[
                "enc_topk_boxes"
            ],

            # Optional debugging/extension outputs
            "topk_indices": query_output["topk_indices"],
            "spatial_sizes": spatial_sizes,
            "level_start_index": level_start_index,
        }
    