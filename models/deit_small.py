"""
Pure Vision Transformer – DeiT-Small – tuned for CIFAR-100 (32×32, patch 4×4).

Architecture matches DeiT-Small from Touvron et al. (2021):
  depth=12  •  embed_dim=384  •  heads=6  •  MLP ratio=4.

Implemented with plain PyTorch (no timm / torchvision dependency)
so training is fully transparent.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and embed them."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0, 'img_size must be divisible by patch_size'
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):                 # B,3,32,32  →  B,384,8,8
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B,64,384
        return x


# -----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features or in_features
        self.fc1   = nn.Linear(in_features, hidden_features)
        self.fc2   = nn.Linear(hidden_features, out_features)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# -----------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim  = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = ( self.qkv(x)
                  .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                  .permute(2, 0, 3, 1, 4) )           # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1   = nn.LayerNorm(dim)
        self.attn    = Attention(dim, num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 attn_drop=attn_drop,
                                 proj_drop=drop)
        self.drop_path = nn.Identity()               # no stochastic depth for small net
        self.norm2   = nn.LayerNorm(dim)
        self.mlp     = MLP(in_features=dim,
                           hidden_features=int(dim * mlp_ratio),
                           drop=drop)

    def forward(self, x):
        x = x + self.drop_path( self.attn( self.norm1(x) ) )
        x = x + self.drop_path( self.mlp ( self.norm2(x) ) )
        return x


# -----------------------------------------------------------
class DeiT_Small_P4_32(nn.Module):
    def __init__(self, num_classes=100, img_size=32, patch_size=4,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 qkv_bias=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size,
                                      in_chans=3, embed_dim=embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(p=0.)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias)
            for _ in range(depth)
        ])
        self.norm   = nn.LayerNorm(embed_dim)
        self.head   = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    # ------------------
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    # ------------------
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                       # B, 64, 384

        cls_tokens = self.cls_token.expand(B, -1, -1) # B, 1, 384
        x = torch.cat((cls_tokens, x), dim=1)         # B, 65, 384
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])
