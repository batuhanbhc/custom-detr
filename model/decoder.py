import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadSelfAttention, MultiScaleDeformableAttention
from .norm import LayerNorm
from .mlp import MLP


class SwiGLUFFN(nn.Module):
    """DEIMv2-faithful SwiGLU FFN (engine/deim/deim_utils.py), itself taken from
    Meta's DINOv2. Hidden dim is half of the vanilla dim_feedforward, not the
    2/3-param-matching convention some other SwiGLU variants use."""

    def __init__(self, d_model, dim_feedforward, dropout=0.0):
        super().__init__()
        hidden_dim = dim_feedforward // 2
        self.w12 = nn.Linear(d_model, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.constant_(self.w12.bias, 0)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.constant_(self.w3.bias, 0)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(x1) * x2))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_levels, k_list, num_class, box_mlp=None, layer_embeddings=None, expansion=4.0, pre_norm = True, ffn_dropout=0.0, msa_dropout=0.0):
        super().__init__()

        self.pre_norm = pre_norm
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, msa_dropout)
        self.cross_attn = MultiScaleDeformableAttention(
            d_model=d_model, 
            num_heads=num_heads,
            num_levels=num_levels,
            k_list=k_list
        )

        hidden_dim = int(d_model * expansion)
        self.ffn = SwiGLUFFN(d_model, hidden_dim, ffn_dropout)
        
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)

        self.box_head = MLP(d_model, d_model, 4, 3)
        self.cls_head = nn.Linear(d_model, num_class)
    
    def forward(
        self,
        tgt,
        query_pos_embed,
        memory,
        reference_box_logits,
        spatial_sizes,
        level_start_index,
        attn_mask=None,
    ):
        reference_boxes = reference_box_logits.sigmoid()

        # Self-attention
        if not self.pre_norm:
            self_out = self.self_attn(
                tgt,
                query_pos_embed,
                attn_mask=attn_mask,
            )
            tgt = self.norm_1(tgt + self_out)
        else:
            self_out = self.self_attn(
                self.norm_1(tgt),
                query_pos_embed,
                attn_mask=attn_mask,
            )
            tgt = tgt + self_out

        # Deformable cross-attention
        if not self.pre_norm:
            cross_out = self.cross_attn(
                tgt,
                query_pos_embed,
                memory,
                reference_boxes,
                spatial_sizes,
                level_start_index,
            )
            tgt = self.norm_2(tgt + cross_out)
        else:
            cross_out = self.cross_attn(
                self.norm_2(tgt),
                query_pos_embed,
                memory,
                reference_boxes,
                spatial_sizes,
                level_start_index,
            )
            tgt = tgt + cross_out

        # FFN
        if not self.pre_norm:
            tgt = self.norm_3(tgt + self.ffn(tgt))
        else:
            tgt = tgt + self.ffn(self.norm_3(tgt))

        box_delta = self.box_head(tgt)
        class_logits = self.cls_head(tgt)

        new_reference_logits = (
            reference_box_logits + box_delta
        )
        pred_boxes = new_reference_logits.sigmoid()

        return {
            "query_tokens": tgt,
            "reference_box_logits": new_reference_logits.detach(),
            "pred_boxes": pred_boxes,
            "box_delta": box_delta,
            "class_logits": class_logits,
        }

class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_levels,
        k_list,
        num_classes,
        num_layers=6,
        expansion=4.0,
        pre_norm=True,
        ffn_dropout=0.0,
        msa_dropout=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers

        # Shared across all decoder layers
        self.query_pos_head = MLP(
            input_dim=4,
            hidden_dim=2 * d_model,
            output_dim=d_model,
            num_layers=2,
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_levels=num_levels,
                k_list=k_list,
                num_class=num_classes,
                expansion=expansion,
                pre_norm=pre_norm,
                ffn_dropout=ffn_dropout,
                msa_dropout=msa_dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        query_tokens,
        memory,
        reference_box_logits,
        spatial_sizes,
        level_start_index,
        attn_mask=None,
    ):
        """
        query_tokens:
            [B, Q, d_model]

        memory:
            [B, total_memory_tokens, d_model]

        reference_box_logits:
            [B, Q, 4], inverse-sigmoid/logit coordinates

        spatial_sizes:
            [num_levels, 2], containing [H_i, W_i]

        level_start_index:
            [num_levels]

        attn_mask:
            optional [Q, Q] or [B, Q, Q]
        """

        tgt = query_tokens
        current_reference_logits = reference_box_logits

        all_class_logits = []
        all_pred_boxes = []
        all_query_tokens = []

        for layer in self.layers:
            current_reference_boxes = current_reference_logits.sigmoid()

            query_pos_embed = self.query_pos_head(
                current_reference_boxes
            )
            # [B, Q, d_model]

            layer_output = layer(
                tgt=tgt,
                query_pos_embed=query_pos_embed,
                memory=memory,
                reference_box_logits=current_reference_logits,
                spatial_sizes=spatial_sizes,
                level_start_index=level_start_index,
                attn_mask=attn_mask,
            )

            tgt = layer_output["query_tokens"]

            # Already detached by DecoderLayer for the next layer.
            current_reference_logits = (
                layer_output["reference_box_logits"]
            )

            all_query_tokens.append(tgt)
            all_class_logits.append(
                layer_output["class_logits"]
            )
            all_pred_boxes.append(
                layer_output["pred_boxes"]
            )

        # [B, num_layers, Q, ...]
        all_query_tokens = torch.stack(
            all_query_tokens,
            dim=1,
        )
        all_class_logits = torch.stack(
            all_class_logits,
            dim=1,
        )
        all_pred_boxes = torch.stack(
            all_pred_boxes,
            dim=1,
        )

        return {
            # Final-layer predictions
            "pred_logits": all_class_logits[:, -1],
            "pred_boxes": all_pred_boxes[:, -1],

            # Used for auxiliary decoder losses
            "all_class_logits": all_class_logits,
            "all_pred_boxes": all_pred_boxes,

            # Potentially useful for later D-FINE-style additions
            "all_query_tokens": all_query_tokens,

            # Refined logit references after the final layer
            "final_reference_box_logits": current_reference_logits,
        }