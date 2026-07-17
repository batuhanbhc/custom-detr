import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model) 
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, pos=None, key_padding_mask=None, attn_mask=None):
        # Assuming x is of shape [batch_size, seq, d_model]
        # pos may be shared [seq, d_model] or batched [batch, seq, d_model].
        # Assuming key_padding_mask is of shape [batch_size, seq]

        batch_size, seq, _ = x.size()
        if pos is not None and pos.dim() == 2:
            pos = pos.unsqueeze(0)  # [1, seq, d_model]

        qk_input = x + pos if pos is not None else x

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v = self.v_proj(x)

        q = q.view(batch_size, seq, self.num_heads, self.d_k).transpose(1,2) # [batch_size, num_heads, seq, d_k]
        k = k.view(batch_size, seq, self.num_heads, self.d_k).transpose(1,2) # [batch_size, num_heads, seq, d_k]
        v = v.view(batch_size, seq, self.num_heads, self.d_k).transpose(1,2) # [batch_size, num_heads, seq, d_k]

        # SDPA selects PyTorch's numerically stable fused implementation when
        # available. Its boolean mask uses True=allowed, the inverse of the
        # project's existing True=blocked masks.
        allowed_mask = None
        if key_padding_mask is not None:
            allowed_mask = ~key_padding_mask[:, None, None, :]

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]
            attention_allowed = ~attn_mask
            allowed_mask = attention_allowed if allowed_mask is None else allowed_mask & attention_allowed

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1,2).reshape(batch_size, seq, self.d_model)
        return self.out_proj(out)
    

class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_levels, k_list):
        super().__init__()

        assert len(k_list) == num_levels
        assert num_levels >= 1
        assert num_heads >= 1
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.k_list = k_list
        self.head_dim = d_model // num_heads

        self.value_proj = nn.Linear(d_model, d_model)

        self.total_k = 0
        for k in k_list:
            self.total_k += k

        self.offset_head = nn.Linear(d_model, num_heads * self.total_k * 2)
        self.attn_head = nn.Linear(d_model, num_heads * self.total_k)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, tgt, query_pos, memory, reference_boxes, spatial_sizes, level_start_index):
        b, num_queries, _ = tgt.size()

        q = tgt + query_pos # [B, num_queries, d_model]
        v = self.value_proj(memory) # [B, seq, d_model]
        v = v.reshape(b, memory.shape[1], self.num_heads, self.head_dim) # [B, seq, heads, head_dim]

        offsets = self.offset_head(q) # [B, num_queries, heads * total_k * 2]
        offsets = offsets.reshape(b, num_queries, self.num_heads, self.total_k, 2)
        attn_logits = self.attn_head(q) # [B, num_queries, heads * total_k]
        attn_logits = attn_logits.reshape(b, num_queries, self.num_heads, self.total_k)
        attn_weights = attn_logits.softmax(dim=-1) # [B, query, head, total_k]

        # reference_boxes is expected to be of size [B, num_queries, 4] in [0-1]
        centers = reference_boxes[..., :2][:, :, None, None, :] # [B, query, 1, 1, 2]
        box_sizes = reference_boxes[..., 2:][:, :, None, None, :] # [B, query, 1, 1, 2]

        offsets_per_map = torch.split(offsets, self.k_list, dim=-2) # each [B, query, head, k_i, 2]
        outs = []
        for level, offset_map in enumerate(offsets_per_map):
            k_i = self.k_list[level]
            h_i, w_i = spatial_sizes[level]

            sampling_offsets = centers + 0.5 * box_sizes * offset_map / k_i # [B, query, head, k_i, 2]
            start_idx = level_start_index[level]
            value_map = v[:, start_idx:start_idx + h_i * w_i, :, :] # [B, h_i*w_i, heads, head_dim]
            value_map = value_map.permute(0, 2, 3, 1) # [B, heads, head_dim, h_i*w_i]
            value_map = value_map.reshape(b * self.num_heads, self.head_dim, h_i, w_i) # [B * heads, head_dim, h_i, w_i]

            grid = sampling_offsets.permute(0, 2, 1, 3, 4) # [B, head, query, k_i, 2]
            grid =  grid.reshape(b * self.num_heads, num_queries, k_i, 2) # [B * head, query, k_i, 2]
            grid = 2 * grid - 1  # [0, 1] to [-1, 1]

            out = F.grid_sample(
                value_map,
                grid, 
                mode="bilinear", 
                padding_mode="zeros",
                align_corners=False
            ) # [B*head, head_dim query, k_i]

            out = out.view(
                b,
                self.num_heads,
                self.head_dim,
                num_queries,
                k_i,
            ).permute(0, 3, 1, 4, 2)
            # [B, num_queries, heads, k_i, head_dim]

            outs.append(out)
        
        attn_values = torch.cat(outs, dim=3) # [B, query, head, total_k, head_dim]

        tgt = attn_weights.unsqueeze(-2) @ attn_values # [B, query, head, 1, head_dim]
        tgt = tgt.squeeze(3).reshape(b, num_queries, self.num_heads * self.head_dim) # [B, query, d_model]
        return self.out_proj(tgt)









