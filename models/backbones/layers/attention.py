# Adapted from timm (PyTorch Image Models) [https://github.com/rwightman/pytorch-image-models]
# Licensed under Apache License 2.0
#
# Modifications:
# - simplified
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep: random_tensor.div_(keep_prob)
        return x * random_tensor

class SelfAttention(nn.Module):
    # QKV Linear -> Multi-Head Self-Attention -> Merge heads -> Linear -> Dropout
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x) # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop) # (B, num_heads, N, head_dim)

        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    # Q(decoder), KV(encoder) Linear -> Multi-Head Cross-Attention -> Merge heads -> Linear -> Dropout
    def __init__(self, encoder_dim, decoder_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, 2 * decoder_dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """query from decoder x, key/value from encoder y."""
        B, N, C = x.shape # (C: decoder_dim)
        q, kv = self.q(x), self.kv(y) # (B, N, C), (B, N, 2*C)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (B, num_heads, N, head_dim)
        kv = kv.reshape(B, y.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (2, B, num_heads, N, head_dim)
        k, v = kv[0], kv[1]
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop) # (B, num_heads, N, head_dim)

        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLPBlock(nn.Module):
    # Linear -> Activation -> Dropout -> Linear -> Dropout
    def __init__(self, in_feats, hidden_feats=None, out_feats=None, activation=nn.GELU, dropout=0.):
        super().__init__()
        out_feats = out_feats or in_feats
        hidden_feats = hidden_feats or in_feats
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.activation = activation()
        self.fc2 = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class SelfAttentionBlock(nn.Module):
    # Residual (Norm -> Self-Attention) -> Residual (Norm -> MLP)
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0., path_drop=0., activation=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(path_drop) if path_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(in_feats=dim, hidden_feats=int(dim * mlp_ratio), activation=activation, dropout=proj_drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttentionBlock(nn.Module):
    # Residual (Norm -> Self-Attention) -> Residual (Norm -> Cross-Attention) -> Residual (Norm -> MLP)
    def __init__(self, encoder_dim, decoder_dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0., path_drop=0., activation=nn.GELU, norm_layer=nn.LayerNorm, self_attn=False):
        super().__init__()
        self.self_attn = self_attn
        if self.self_attn:
            self.norm0 = norm_layer(decoder_dim)
            self.self_attn = SelfAttention(decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm1 = norm_layer(decoder_dim)
        self.cross_attn = CrossAttention(encoder_dim, decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(path_drop) if path_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(decoder_dim)
        self.mlp = MLPBlock(in_feats=decoder_dim, hidden_feats=int(decoder_dim * mlp_ratio), activation=activation, dropout=proj_drop)

    def forward(self, x, y):
        """query from decoder x, key/value from encoder y."""
        if self.self_attn: x = x + self.drop_path(self.self_attn(self.norm0(x)))
        x = x + self.drop_path(self.cross_attn(self.norm1(x), y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
