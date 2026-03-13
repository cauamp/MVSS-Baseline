"""
RTMVSS-6: SAM2-Based Real-Time Multispectral Video Semantic Segmentation Model

Architecture overview:
  1. Dual-stream HiERA backbone (SAM2-inspired Hierarchical Vision Transformer)
       - Four stages: 1/4, 1/8, 1/16, 1/32 resolution
       - Window-based self-attention for efficiency; global attention at deep stages
       - Relative positional biases within each attention window
  2. Cross-modal attention fusion (RGB ↔ IR) at multiple feature scales
  3. SAM2-style key-value memory attention for temporal video context
  4. FPN-style multi-scale decoder (top-down + lateral connections)
  5. Multi-class segmentation head with per-modality auxiliary outputs

Improvements over the MVNet baseline:
  - Hierarchical ViT backbone (vs ResNet-50 + ASPP)
  - Cross-modal cross-attention fusion (vs simple concatenation)
  - Attention-based temporal memory bank (vs class-prototype memory)
  - FPN decoder leverages multi-scale context (vs DeepLabv3+ single-scale decoder)
  - DropPath + LayerScale regularisation for improved generalisation

Input/output interface is fully compatible with the existing MVNet training loop:
  forward(rgb_seq, ir_seq, step=0, epoch=0)
  → (output, aux_rgb, aux_thermal, aux_fusion, total_feas)
"""

import logging
import math
import random
import warnings
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 4-D tensors (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth / DropPath regularisation."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # Bernoulli mask: 1 = keep, 0 = drop; scale kept values to maintain expectation
    mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
    return x.div(keep_prob) * mask.to(x.dtype)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Patch Embedding (overlapping, 4× downsampling via two stride-2 convolutions)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Overlapping patch embedding that maps (B, in_chans, H, W) →
    (B, embed_dim, H/4, W/4) using two stride-2 convolutions separated by
    a channel-first LayerNorm.
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Window-based multi-head self-attention with relative positional biases
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA) with relative positional
    biases, as used in Swin Transformer and SAM2's HiERA backbone.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative positional bias table: (2Wh-1) × (2Ww-1) entries
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        # Pre-compute relative position indices
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flat = torch.flatten(coords, 1)  # 2, Wh*Ww
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, N, N
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        rel_idx = rel.sum(-1)  # N, N
        self.register_buffer("rel_idx", rel_idx)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_windows*B, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative positional bias
        rel_bias = self.rel_bias_table[self.rel_idx.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # N, N, num_heads
        rel_bias = rel_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # 1, heads, N, N
        attn = attn + rel_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Window partition utilities
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition feature map into non-overlapping windows."""
    B, H, W, C = x.shape
    # Pad if necessary
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows.view(-1, window_size * window_size, C), (Hp, Wp)


def window_reverse(windows: torch.Tensor, window_size: int, Hp: int, Wp: int, H: int, W: int) -> torch.Tensor:
    """Reconstruct feature map from windows."""
    B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# ---------------------------------------------------------------------------
# HiERA Transformer Block (SAM2-inspired)
# ---------------------------------------------------------------------------

class HiERABlock(nn.Module):
    """
    Hierarchical Encoder with Relative-position-biased Attention (HiERA) block.
    Uses windowed attention by default; set window_size=0 for global attention.
    Includes LayerScale and DropPath for training stability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        ws = (window_size, window_size) if window_size > 0 else (1, 1)
        self.attn = WindowAttention(
            dim, window_size=ws, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  channel-first
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        shortcut = x_seq
        x_seq = self.norm1(x_seq)

        if self.window_size > 0:
            # Window attention
            x_win, (Hp, Wp) = window_partition(x_seq, self.window_size)
            x_win = self.attn(x_win)
            x_seq = window_reverse(x_win, self.window_size, Hp, Wp, H, W)
        else:
            # Global attention: flatten spatial dims
            x_flat = x_seq.view(B, H * W, C)
            x_flat = self.attn(x_flat)
            x_seq = x_flat.view(B, H, W, C)

        x_seq = shortcut + self.drop_path(self.gamma1 * x_seq)
        x_seq = x_seq + self.drop_path(self.gamma2 * self.mlp(self.norm2(x_seq)))
        return x_seq.permute(0, 3, 1, 2)  # back to (B, C, H, W)


# ---------------------------------------------------------------------------
# Hierarchical Downsampling
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """2× spatial downsampling via strided convolution (channel doubling)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(self.norm(x))


# ---------------------------------------------------------------------------
# Hierarchical Encoder (SAM2-style HiERA backbone)
# ---------------------------------------------------------------------------

class HiERAEncoder(nn.Module):
    """
    Four-stage hierarchical ViT encoder inspired by SAM2's HiERA backbone.

    Stage resolutions (relative to input):
      Stage 0: 1/4   (after PatchEmbed)
      Stage 1: 1/8   (after PatchMerging)
      Stage 2: 1/16  (after PatchMerging)
      Stage 3: 1/32  (after PatchMerging)

    Args:
        in_chans:    Input channels (3 for RGB, 3 for IR replicated to match)
        embed_dim:   Base embedding dimension (doubles each stage)
        depths:      Number of transformer blocks per stage
        num_heads:   Number of attention heads per stage
        window_size: Attention window size (0 = global attention)
        mlp_ratio:   MLP expansion ratio
        drop_path_rate: Stochastic depth decay rate
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        global_attn_stages: Tuple[int, ...] = (3,),
    ):
        super().__init__()
        self.num_stages = len(depths)
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        # Stochastic depth schedule
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0

        self.patch_embed = PatchEmbed(in_chans, embed_dim)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for stage_idx in range(self.num_stages):
            ws = 0 if stage_idx in global_attn_stages else window_size
            stage_blocks = nn.Sequential(*[
                HiERABlock(
                    dim=dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    window_size=ws,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[block_idx + i],
                )
                for i in range(depths[stage_idx])
            ])
            self.stages.append(stage_blocks)
            block_idx += depths[stage_idx]

            if stage_idx < self.num_stages - 1:
                self.downsamples.append(PatchMerging(dims[stage_idx], dims[stage_idx + 1]))

        self.out_dims = dims  # [96, 192, 384, 768] for embed_dim=96

        # Per-stage output norms
        self.norms = nn.ModuleList([LayerNorm2d(d) for d in dims])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of 4 feature maps at 1/4, 1/8, 1/16, 1/32 resolution."""
        x = self.patch_embed(x)
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(self.norms[i](x))
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)
        return features


# ---------------------------------------------------------------------------
# Cross-Modal Attention Fusion
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention between RGB and IR feature maps.
    Query comes from one modality; Key/Value from the other.
    Uses lightweight convolution projections for speed.
    """

    def __init__(self, dim: int, num_heads: int = 8, reduction: int = 4):
        super().__init__()
        inner = max(dim // reduction, 32)
        # Ensure inner is divisible by num_heads; pick largest valid num_heads
        original_heads = num_heads
        for nh in [num_heads, 8, 4, 2, 1]:
            if inner % nh == 0:
                num_heads = nh
                break
        if num_heads != original_heads:
            warnings.warn(
                f"CrossModalAttention: requested num_heads={original_heads} does not evenly divide "
                f"inner_dim={inner}; using num_heads={num_heads} instead.",
                UserWarning,
            )
        self.num_heads = num_heads
        self.head_dim = inner // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_rgb = nn.Conv2d(dim, inner, 1)
        self.kv_ir = nn.Conv2d(dim, inner * 2, 1)
        self.q_ir = nn.Conv2d(dim, inner, 1)
        self.kv_rgb = nn.Conv2d(dim, inner * 2, 1)

        self.proj_rgb = nn.Sequential(
            nn.Conv2d(inner, dim, 1),
            LayerNorm2d(dim),
        )
        self.proj_ir = nn.Sequential(
            nn.Conv2d(inner, dim, 1),
            LayerNorm2d(dim),
        )

        # Gating: learn how much cross-modal information to admit
        self.gate_rgb = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())
        self.gate_ir = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())

        self.norm_rgb = LayerNorm2d(dim)
        self.norm_ir = LayerNorm2d(dim)

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        B, C, _, _ = q.shape
        nh = self.num_heads
        hd = C // nh

        # Flatten spatial to sequence
        q = q.flatten(2).permute(0, 2, 1).view(B, H * W, nh, hd).permute(0, 2, 1, 3)
        k = k.flatten(2).permute(0, 2, 1).view(B, H * W, nh, hd).permute(0, 2, 1, 3)
        v = v.flatten(2).permute(0, 2, 1).view(B, H * W, nh, hd).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, H * W, C)
        return out.permute(0, 2, 1).view(B, C, H, W)

    def forward(
        self, f_rgb: torch.Tensor, f_ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = f_rgb.shape

        # RGB attends to IR
        q_r = self.q_rgb(self.norm_rgb(f_rgb))
        kv_i = self.kv_ir(self.norm_ir(f_ir))
        k_i, v_i = kv_i.chunk(2, dim=1)
        rgb_cross = self._attn(q_r, k_i, v_i, H, W)
        rgb_cross = self.proj_rgb(rgb_cross)
        gate_r = self.gate_rgb(torch.cat([f_rgb, rgb_cross], dim=1))
        f_rgb_out = f_rgb + gate_r * rgb_cross

        # IR attends to RGB
        q_i = self.q_ir(self.norm_ir(f_ir))
        kv_r = self.kv_rgb(self.norm_rgb(f_rgb))
        k_r, v_r = kv_r.chunk(2, dim=1)
        ir_cross = self._attn(q_i, k_r, v_r, H, W)
        ir_cross = self.proj_ir(ir_cross)
        gate_i = self.gate_ir(torch.cat([f_ir, ir_cross], dim=1))
        f_ir_out = f_ir + gate_i * ir_cross

        return f_rgb_out, f_ir_out


# ---------------------------------------------------------------------------
# SAM2-Inspired Memory Attention
# ---------------------------------------------------------------------------

class MemoryEncoder(nn.Module):
    """
    Compresses current-frame features into a (key, value) pair for the memory
    bank.  A lightweight convolution first projects the features; keys and
    values are then produced by separate 1×1 convolutions.
    """

    def __init__(self, in_dim: int, mem_dim: int = 64):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_dim, mem_dim, 1, bias=False),
            LayerNorm2d(mem_dim),
            nn.GELU(),
        )
        self.key_proj = nn.Conv2d(mem_dim, mem_dim, 1)
        self.val_proj = nn.Conv2d(mem_dim, mem_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.compress(x)
        return self.key_proj(z), self.val_proj(z)


class MemoryAttention(nn.Module):
    """
    SAM2-style memory attention module.

    At each time step the current-frame features attend to a bank of (key,
    value) pairs drawn from previous frames.  A learnable query projection maps
    the current features to queries; the retrieved memory is blended back via
    a gated residual.

    Memory bank layout:  keys/values stored as lists of (B, mem_dim, H', W')
    tensors; up to ``max_mem`` entries are kept (FIFO).
    """

    def __init__(self, in_dim: int, mem_dim: int = 64, num_heads: int = 4, max_mem: int = 5):
        super().__init__()
        self.mem_dim = mem_dim
        self.num_heads = num_heads
        self.head_dim = mem_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_mem = max_mem

        self.query_proj = nn.Sequential(
            nn.Conv2d(in_dim, mem_dim, 1, bias=False),
            LayerNorm2d(mem_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(mem_dim, in_dim, 1, bias=False),
            LayerNorm2d(in_dim),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Bank management (stateful, reset per video clip)
    # ------------------------------------------------------------------

    def init_bank(self):
        self._key_bank: deque = deque(maxlen=self.max_mem)
        self._val_bank: deque = deque(maxlen=self.max_mem)

    def update_bank(self, key: torch.Tensor, val: torch.Tensor):
        """Add a new (key, value) pair; oldest entry is evicted automatically."""
        self._key_bank.append(key)
        self._val_bank.append(val)

    def bank_size(self) -> int:
        return len(self._key_bank)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: current frame features (B, in_dim, H, W)
        Returns:
            x enriched with temporal memory context, same shape
        """
        if self.bank_size() == 0:
            return x  # no memory yet – pass through unchanged

        B, C, H, W = x.shape
        nh, hd = self.num_heads, self.head_dim

        # Query from current frame
        q = self.query_proj(x)  # (B, mem_dim, H, W)
        q = q.flatten(2).permute(0, 2, 1)  # (B, HW, mem_dim)
        q = q.view(B, H * W, nh, hd).permute(0, 2, 1, 3)  # (B, nh, HW, hd)

        # Keys and values from memory bank
        # Each entry: (B, mem_dim, Hm, Wm)
        keys = torch.stack(list(self._key_bank), dim=2)  # (B, mem_dim, T, Hm, Wm)
        vals = torch.stack(list(self._val_bank), dim=2)
        T, Hm, Wm = keys.shape[2], keys.shape[3], keys.shape[4]

        # Flatten memory spatial + temporal dims
        k = keys.permute(0, 2, 3, 4, 1).reshape(B, T * Hm * Wm, self.mem_dim)
        v = vals.permute(0, 2, 3, 4, 1).reshape(B, T * Hm * Wm, self.mem_dim)

        k = k.view(B, -1, nh, hd).permute(0, 2, 1, 3)  # (B, nh, TN, hd)
        v = v.view(B, -1, nh, hd).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, HW, TN)
        attn = attn.softmax(dim=-1)
        mem_out = (attn @ v).permute(0, 2, 1, 3).reshape(B, H * W, self.mem_dim)
        mem_out = mem_out.permute(0, 2, 1).view(B, self.mem_dim, H, W)
        mem_out = self.out_proj(mem_out)

        gate = self.gate(torch.cat([x, mem_out], dim=1))
        return x + gate * mem_out


# ---------------------------------------------------------------------------
# FPN-Style Multi-Scale Decoder
# ---------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network (FPN) decoder with top-down pathway.

    Takes four feature maps (1/4, 1/8, 1/16, 1/32) and fuses them top-down,
    producing a single feature map at 1/4 resolution.
    """

    def __init__(self, in_dims: List[int], out_dim: int = 256):
        super().__init__()
        # Lateral convolutions (reduce each stage to out_dim)
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, out_dim, 1, bias=False),
                LayerNorm2d(out_dim),
            )
            for d in in_dims
        ])
        # Top-down fusion convolutions
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
                LayerNorm2d(out_dim),
                nn.GELU(),
            )
            for _ in range(len(in_dims) - 1)
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: list of 4 tensors at [1/4, 1/8, 1/16, 1/32] resolution.
        Returns:  single tensor at 1/4 resolution.
        """
        # Apply lateral projections
        laterals = [lat(f) for lat, f in zip(self.lateral, features)]

        # Top-down merge: start from the coarsest (index 3) and go up
        out = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            out = F.interpolate(out, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            out = laterals[i] + out
            out = self.fuse[i](out)

        return out  # (B, out_dim, H/4, W/4)


# ---------------------------------------------------------------------------
# Segmentation Head
# ---------------------------------------------------------------------------

class SegHead(nn.Module):
    """Dense segmentation head with two refinement convolutions."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.conv(x)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Multimodal Fusion at each FPN level
# ---------------------------------------------------------------------------

class ModalFuse(nn.Module):
    """
    Fuses RGB and IR FPN features into a single representation.
    Uses a channel-attention (SE-like) mechanism before element-wise addition.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim * 2),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            LayerNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, f_rgb: torch.Tensor, f_ir: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([f_rgb, f_ir], dim=1)
        w = self.se(cat).view(cat.shape[0], -1, 1, 1)
        return self.proj(cat * w)


# ---------------------------------------------------------------------------
# Memory Queue (state management per video clip)
# ---------------------------------------------------------------------------

class MemoryBank:
    """Simple FIFO memory bank wrapping three MemoryAttention modules."""

    def __init__(self):
        self.rgb_attn: Optional[MemoryAttention] = None
        self.ir_attn: Optional[MemoryAttention] = None
        self.fused_attn: Optional[MemoryAttention] = None

    def register(self, rgb_attn: "MemoryAttention", ir_attn: "MemoryAttention", fused_attn: "MemoryAttention"):
        self.rgb_attn = rgb_attn
        self.ir_attn = ir_attn
        self.fused_attn = fused_attn

    def reset(self):
        for m in (self.rgb_attn, self.ir_attn, self.fused_attn):
            if m is not None:
                m.init_bank()


# ---------------------------------------------------------------------------
# Main Model: RTMVSS-6
# ---------------------------------------------------------------------------

class RTMVSS6(nn.Module):
    """
    RTMVSS-6: SAM2-Based Real-Time Multispectral Video Semantic Segmentation.

    Args:
        num_classes:      Number of semantic classes (default: 26 for MVSeg).
        embed_dim:        Base channel dimension of HiERA backbone (default: 96).
        depths:           Number of transformer blocks per stage.
        num_heads:        Attention heads per stage.
        window_size:      Attention window size for windowed stages.
        fpn_dim:          Channel dimension of FPN feature maps.
        mem_dim:          Channel dimension of memory bank keys/values.
        max_mem:          Maximum number of frames stored in memory.
        drop_path_rate:   Maximum stochastic depth probability.
        share_backbone:   If True, RGB and IR streams share backbone weights.
        stm_queue_size:   Alias for max_mem (for compatibility with MVNet args).
        sample_rate:      Frame sampling strategy (unused; for compatibility).
        memory_strategy:  Memory update strategy ("all" or "random").
        always_decode:    Whether to produce predictions for every frame.
        baseline_mode:    Disable memory when True (ablation).
    """

    def __init__(
        self,
        num_classes: int = 26,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        fpn_dim: int = 256,
        mem_dim: int = 64,
        max_mem: int = 5,
        drop_path_rate: float = 0.1,
        share_backbone: bool = False,
        # Compatibility kwargs (may be passed as an args namespace)
        stm_queue_size: int = 5,
        sample_rate: int = 1,
        memory_strategy: str = "all",
        always_decode: bool = False,
        baseline_mode: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_mem = max(max_mem, stm_queue_size)
        self.memory_strategy = memory_strategy
        self.always_decode = always_decode
        self.baseline_mode = baseline_mode
        self.share_backbone = share_backbone

        # ---- Backbone ----
        backbone_kwargs = dict(
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
        )
        self.backbone_rgb = HiERAEncoder(**backbone_kwargs)
        if share_backbone:
            self.backbone_ir = self.backbone_rgb
        else:
            self.backbone_ir = HiERAEncoder(**backbone_kwargs)

        stage_dims = self.backbone_rgb.out_dims  # e.g. [96, 192, 384, 768]

        # ---- Cross-modal attention at each scale ----
        self.cross_modal = nn.ModuleList([
            CrossModalAttention(d, num_heads=min(d // 32, 8) or 1)
            for d in stage_dims
        ])

        # ---- Memory encoder (operates on bottleneck features, stage 3) ----
        bottleneck_dim = stage_dims[-1]
        self.mem_encoder_rgb = MemoryEncoder(bottleneck_dim, mem_dim)
        self.mem_encoder_ir = MemoryEncoder(bottleneck_dim, mem_dim)
        self.mem_encoder_fused = MemoryEncoder(bottleneck_dim, mem_dim)

        # ---- Memory attention modules ----
        self.mem_attn_rgb = MemoryAttention(bottleneck_dim, mem_dim, num_heads=4, max_mem=self.max_mem)
        self.mem_attn_ir = MemoryAttention(bottleneck_dim, mem_dim, num_heads=4, max_mem=self.max_mem)
        self.mem_attn_fused = MemoryAttention(bottleneck_dim, mem_dim, num_heads=4, max_mem=self.max_mem)

        # ---- Modality fusion before FPN ----
        self.modal_fuse = nn.ModuleList([ModalFuse(d) for d in stage_dims])

        # ---- FPN decoders (one per modality stream) ----
        self.fpn_rgb = FPNDecoder(stage_dims, fpn_dim)
        self.fpn_ir = FPNDecoder(stage_dims, fpn_dim)
        self.fpn_fused = FPNDecoder(stage_dims, fpn_dim)

        # ---- Segmentation heads ----
        self.seg_head_rgb = SegHead(fpn_dim, num_classes)
        self.seg_head_ir = SegHead(fpn_dim, num_classes)
        self.seg_head_fused = SegHead(fpn_dim, num_classes)
        # Final ensemble head
        self.seg_head_final = nn.Conv2d(num_classes * 3, num_classes, 1)

        # ---- Memory bank state ----
        self._mem_bank = MemoryBank()
        self._mem_bank.register(self.mem_attn_rgb, self.mem_attn_ir, self.mem_attn_fused)
        self.reset_hidden_state()

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_hidden_state(self):
        """Reset memory banks (call at the start of each new video clip)."""
        self._mem_bank.reset()

    # ------------------------------------------------------------------
    # Memory range selection
    # ------------------------------------------------------------------

    def _memory_range(self, seq_len: int) -> List[int]:
        if self.memory_strategy == "random":
            if seq_len <= 1:
                return list(range(seq_len))
            # Select up to (max_mem - 1) context frames from all but the last frame,
            # then always include the final (query) frame.
            n_context = min(self.max_mem - 1, seq_len - 1)
            r = random.sample(range(seq_len - 1), n_context)
            r.append(seq_len - 1)
            return sorted(r)
        return list(range(seq_len))  # "all"

    # ------------------------------------------------------------------
    # Single-frame encode → fuse → decode
    # ------------------------------------------------------------------

    def _encode_frame(
        self, rgb: torch.Tensor, ir: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Extract and cross-fuse multi-scale features for one frame."""
        feats_rgb = self.backbone_rgb(rgb)
        feats_ir = self.backbone_ir(ir)

        # Cross-modal attention at each scale
        fused_feats_rgb, fused_feats_ir = [], []
        for i, (fr, fi, cm) in enumerate(zip(feats_rgb, feats_ir, self.cross_modal)):
            fr2, fi2 = cm(fr, fi)
            fused_feats_rgb.append(fr2)
            fused_feats_ir.append(fi2)

        # Modality-fused features (for the fusion stream)
        fused_feats = [self.modal_fuse[i](fused_feats_rgb[i], fused_feats_ir[i])
                       for i in range(len(feats_rgb))]

        return fused_feats_rgb, fused_feats_ir, fused_feats

    def _apply_memory(
        self,
        feats_rgb: List[torch.Tensor],
        feats_ir: List[torch.Tensor],
        feats_fused: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Apply memory attention to the bottleneck (deepest) feature map.

        Only the bottleneck (last) element is affected by memory; shallow feature
        maps are passed through unchanged.  We clone only the bottleneck to
        avoid modifying the caller's tensors while minimising memory overhead.
        """
        f_rgb = feats_rgb[:-1] + [self.mem_attn_rgb(feats_rgb[-1].clone())]
        f_ir = feats_ir[:-1] + [self.mem_attn_ir(feats_ir[-1].clone())]
        f_fused = feats_fused[:-1] + [self.mem_attn_fused(feats_fused[-1].clone())]
        return f_rgb, f_ir, f_fused

    def _decode_frame(
        self,
        feats_rgb: List[torch.Tensor],
        feats_ir: List[torch.Tensor],
        feats_fused: List[torch.Tensor],
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run FPN decoders and segmentation heads for one frame."""
        fpn_r = self.fpn_rgb(feats_rgb)
        fpn_i = self.fpn_ir(feats_ir)
        fpn_f = self.fpn_fused(feats_fused)

        pred_rgb = self.seg_head_rgb(fpn_r, h, w)
        pred_ir = self.seg_head_ir(fpn_i, h, w)
        pred_fused = self.seg_head_fused(fpn_f, h, w)

        # Final ensemble: concatenate and project
        pred_final = self.seg_head_final(
            torch.cat([pred_rgb, pred_ir, pred_fused], dim=1)
        )

        return pred_final, pred_rgb, pred_ir, pred_fused

    def _memorise_frame(
        self,
        feats_rgb: List[torch.Tensor],
        feats_ir: List[torch.Tensor],
        feats_fused: List[torch.Tensor],
    ):
        """Encode current bottleneck features and push to memory bank."""
        k_r, v_r = self.mem_encoder_rgb(feats_rgb[-1])
        k_i, v_i = self.mem_encoder_ir(feats_ir[-1])
        k_f, v_f = self.mem_encoder_fused(feats_fused[-1])

        self.mem_attn_rgb.update_bank(k_r, v_r)
        self.mem_attn_ir.update_bank(k_i, v_i)
        self.mem_attn_fused.update_bank(k_f, v_f)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        rgb_seq: torch.Tensor,
        ir_seq: torch.Tensor,
        step: int = 0,
        epoch: int = 0,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[list],
    ]:
        """
        Args:
            rgb_seq:  (B, T, 3, H, W) visible-spectrum frames.
            ir_seq:   (B, T, 3, H, W) infrared/thermal frames.
            step:     Global training step (unused directly; kept for compat).
            epoch:    Training epoch (unused directly; kept for compat).

        Returns:
            output_decoder:       (B, T_out, num_classes, H, W)
            outputs_aux_rgb:      (B, T_out, num_classes, H, W) or None
            outputs_aux_thermal:  (B, T_out, num_classes, H, W) or None
            outputs_aux_fusion:   (B, T_out, num_classes, H, W) or None
            total_feas:           None (memory loss not used in this version)
        """
        B, T = rgb_seq.shape[:2]
        H, W = int(rgb_seq.shape[3]), int(rgb_seq.shape[4])

        mem_range = self._memory_range(T)

        preds, aux_rgb, aux_ir, aux_fused = [], [], [], []

        for t in mem_range:
            rgb_t = rgb_seq[:, t]
            ir_t = ir_seq[:, t]

            # Encode
            feats_rgb, feats_ir, feats_fused = self._encode_frame(rgb_t, ir_t)

            if self.baseline_mode:
                # No memory
                if not (self.always_decode or t == T - 1):
                    continue
                pred, p_r, p_i, p_f = self._decode_frame(feats_rgb, feats_ir, feats_fused, H, W)
                preds.append(pred)
                aux_rgb.append(p_r)
                aux_ir.append(p_i)
                aux_fused.append(p_f)
                continue

            # Apply memory attention then decode for the *query* frame
            if self.always_decode or t == T - 1:
                f_r, f_i, f_f = self._apply_memory(feats_rgb, feats_ir, feats_fused)
                pred, p_r, p_i, p_f = self._decode_frame(f_r, f_i, f_f, H, W)
                preds.append(pred)
                aux_rgb.append(p_r)
                aux_ir.append(p_i)
                aux_fused.append(p_f)

            # Memorise frames: context frames always, query frame too so it can
            # serve as context for future windows (important in streaming mode).
            if not self.baseline_mode:
                self._memorise_frame(feats_rgb, feats_ir, feats_fused)

        output_decoder = torch.stack(preds, dim=1)
        outputs_aux_rgb = torch.stack(aux_rgb, dim=1) if aux_rgb else None
        outputs_aux_thermal = torch.stack(aux_ir, dim=1) if aux_ir else None
        outputs_aux_fusion = torch.stack(aux_fused, dim=1) if aux_fused else None

        return output_decoder, outputs_aux_rgb, outputs_aux_thermal, outputs_aux_fusion, None


# ---------------------------------------------------------------------------
# Factory helper – build from MVNet-style args namespace
# ---------------------------------------------------------------------------

def build_rtmvss6_from_args(args) -> RTMVSS6:
    """Construct RTMVSS6 from an argparse Namespace compatible with MVNet."""
    # Map dataset → num_classes
    from datasets.helpers import DATASETS_NUM_CLASSES
    num_classes = DATASETS_NUM_CLASSES[args.dataset]

    # Size variant controlled by backbone name
    variant = getattr(args, "backbone", "hiera_tiny")
    if "small" in variant:
        embed_dim, depths, heads = 96, (2, 2, 9, 2), (3, 6, 12, 24)
    elif "base" in variant or "sam2" in variant:
        embed_dim, depths, heads = 112, (2, 3, 16, 3), (4, 8, 16, 16)
    elif "large" in variant:
        embed_dim, depths, heads = 144, (2, 6, 36, 4), (6, 12, 24, 24)
    else:  # default / tiny
        embed_dim, depths, heads = 96, (2, 2, 6, 2), (3, 6, 12, 24)

    return RTMVSS6(
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=heads,
        window_size=getattr(args, "window_size", 7),
        fpn_dim=256,
        mem_dim=64,
        max_mem=getattr(args, "stm_queue_size", 5),
        drop_path_rate=getattr(args, "drop_path_rate", 0.1),
        share_backbone=getattr(args, "share_backbone", False),
        memory_strategy=getattr(args, "memory_strategy", "all"),
        always_decode=getattr(args, "always_decode", False),
        baseline_mode=getattr(args, "baseline_mode", False),
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MVSeg")
    parser.add_argument("--backbone", default="hiera_tiny")
    parser.add_argument("--stm-queue-size", type=int, default=3)
    parser.add_argument("--memory-strategy", default="all")
    parser.add_argument("--always-decode", action="store_true")
    parser.add_argument("--baseline-mode", action="store_true")
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--share-backbone", action="store_true")
    args = parser.parse_args()

    model = RTMVSS6(
        num_classes=26,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        stm_queue_size=args.stm_queue_size,
        memory_strategy=args.memory_strategy,
        always_decode=args.always_decode,
        baseline_mode=args.baseline_mode,
    )
    model.eval()

    B, T, C, H, W = 2, 4, 3, 320, 480
    rgb = torch.randn(B, T, C, H, W)
    ir = torch.randn(B, T, C, H, W)

    with torch.no_grad():
        out, aux_r, aux_i, aux_f, _ = model(rgb, ir)

    print(f"Output shape:      {out.shape}")         # (2, 1, 26, 320, 480)
    print(f"Aux RGB shape:     {aux_r.shape}")
    print(f"Aux IR shape:      {aux_i.shape}")
    print(f"Aux fusion shape:  {aux_f.shape}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f} M")
