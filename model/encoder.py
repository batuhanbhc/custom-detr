import copy
import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .norm import LayerNorm
from .fusion import SimpleELAN
from .pos_embed import build_2d_sin_cos_embed

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, pre_norm=False, expansion=4.0, ffn_dropout=0.0, msa_dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.pre_norm = pre_norm

        self.msa = MultiHeadSelfAttention(d_model, num_heads, msa_dropout)

        hidden_dim = int(d_model * expansion)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(hidden_dim, d_model),
        )

        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
    
    def forward(self, x, pos):
        if self.pre_norm:
            x = self.msa(self.norm_1(x), pos) + x
        else:
            x = self.norm_1(x + self.msa(x, pos))

        if self.pre_norm:
            x = self.ffn(self.norm_2(x)) + x
        else:
            x = self.norm_2(self.ffn(x) + x)
        
        return x
    
class AIFI(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, pre_norm=False, expansion=4.0, ffn_dropout=0.0, msa_dropout=0.0):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        encoder_layer = EncoderLayer(d_model, num_heads, pre_norm, expansion, ffn_dropout, msa_dropout)
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, x):
        batch_size, c, h, w = x.size()

        assert c == self.d_model
        
        pos_embed = build_2d_sin_cos_embed(
            height=h,
            width=w,
            embed_dim=self.d_model,
            temperature=10000.0,
            device=x.device,
            dtype=x.dtype,
        )

        x = x.flatten(2).transpose(1, 2)
        for layer in self.encoder_layers:
            x = layer(x, pos_embed)

        return x.transpose(1,2).reshape(batch_size, c, h, w)
    

class HybridEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8,
                num_enc_layers=1, pre_norm=False, expansion=4, ffn_dropout=0.0, msa_dropout=0.0):
        super().__init__()
        
        self.aifi = AIFI(d_model, num_heads, num_enc_layers, pre_norm, expansion, ffn_dropout, msa_dropout)

        self.lateral_proj_1 = nn.Sequential(
            nn.Conv2d(512, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
        )
        self.lateral_proj_2 = nn.Sequential(
            nn.Conv2d(1024, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
        )
        self.lateral_proj_3 = nn.Sequential(
            nn.Conv2d(2048, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.downsample_1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )

        self.downsample_2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )

        self.fuse_up_1 = SimpleELAN(2*d_model, d_model, num_blocks=2)
        self.fuse_up_2 = SimpleELAN(2*d_model, d_model, num_blocks=2)
        self.fuse_down_1 = SimpleELAN(2*d_model, d_model, num_blocks=2)
        self.fuse_down_2 = SimpleELAN(2*d_model, d_model, num_blocks=2)

    def forward(self, features):
        s3, s4, s5 = features

        p3_lateral = self.lateral_proj_1(s3)
        p4_lateral = self.lateral_proj_2(s4)
        p5_lateral = self.lateral_proj_3(s5)

        p5_encoded = self.aifi(p5_lateral)

        # Top-down path
        p4_top_down = self.fuse_up_1(
            torch.cat(
                [p4_lateral, self.upsample(p5_encoded)],
                dim=1,
            )
        )

        p3_out = self.fuse_up_2(
            torch.cat(
                [p3_lateral, self.upsample(p4_top_down)],
                dim=1,
            )
        )

        # Bottom-up path
        p4_out = self.fuse_down_1(
            torch.cat(
                [p4_top_down, self.downsample_1(p3_out)],
                dim=1,
            )
        )

        p5_out = self.fuse_down_2(
            torch.cat(
                [p5_encoded, self.downsample_2(p4_out)],
                dim=1,
            )
        )

        return p3_out, p4_out, p5_out
    