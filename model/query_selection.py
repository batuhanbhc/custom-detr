import torch
import torch.nn as nn
from .norm import LayerNorm
from .mlp import MLP

class UncertaintyMinimalQuery(nn.Module):
    def __init__(self, d_model, num_class, k=300, base_w=0.05, base_h=0.05):
        super().__init__()

        self.k = k
        self.box_head = MLP(d_model, d_model, 4, 3)
        self.class_head = nn.Linear(d_model, num_class)
        self.base_w = base_w
        self.base_h = base_h

        self.memory_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            LayerNorm(d_model),
        )
    
    def forward(self, features):
        # features is of shape [p3, p4, p5], where each map is
        # [B, d_model, H_i, W_i]
        b,d_model,_,_ = features[0].size()

        # Calculate anchor box reference map in logits
        anchors = []
        for level, feat in enumerate(features):
            _, _, h_i, w_i = feat.size()
            scale = 2**level
            cy = (torch.arange(0, h_i, 1, device=feat.device, dtype=feat.dtype) + 0.5) / h_i # [h_i]
            cx = (torch.arange(0, w_i, 1, device=feat.device, dtype=feat.dtype) + 0.5) / w_i # [w_i]

            cx = cx[None, :, None].expand(h_i, -1, 1) # [h_i, w_i, 1]
            cy = cy[:, None, None].expand(-1, w_i, 1) # [h_i, w_i, 1]
            w = torch.ones(h_i, w_i, 1, device=feat.device, dtype=feat.dtype) * (self.base_w * scale) 
            h = torch.ones(h_i, w_i, 1, device=feat.device, dtype=feat.dtype) * (self.base_h * scale)

            anchors_map_i = torch.cat([cx, cy, w, h], dim=2).unsqueeze(0)  # [1, h_i, w_i, 4]
            anchors.append(anchors_map_i.reshape(1, h_i*w_i, 4))
        
        anchor_logits = torch.logit(
            torch.cat(anchors, dim=1).expand(b, -1, 4),
            eps=1e-6
        )
        
        flat_memory = torch.cat(
            [feat.flatten(2).transpose(1, 2) for feat in features], dim=1
        )   # [B, seq, d_model]

        memory_output = self.memory_proj(flat_memory) # [B, seq, d_model]

        enc_box_logits = self.box_head(memory_output) + anchor_logits # [B, seq, 4]
        enc_class_logits = self.class_head(memory_output)   # [B, seq, nc]

        confidence_scores = torch.max(enc_class_logits, dim=-1).values # [B, seq]
        topk_indices = torch.topk(confidence_scores, self.k).indices # [B, K]

        query_tokens = torch.gather(
            memory_output,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(
                -1, -1, memory_output.shape[-1]
            ),
        )

        topk_class_logits = torch.gather(
            enc_class_logits,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(
                -1, -1, enc_class_logits.shape[-1]
            ),
        )

        topk_box_logits = torch.gather(
            enc_box_logits,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, 4),
        )

        topk_boxes = topk_box_logits.sigmoid()

        return {
            "memory": memory_output,

            # Decoder initialization: detached
            "query_tokens": query_tokens.detach(),
            "reference_box_logits": topk_box_logits.detach(),

            # Encoder auxiliary supervision: not detached
            "enc_topk_class_logits": topk_class_logits,
            "enc_topk_boxes": topk_boxes,

            "topk_indices": topk_indices,
        }
