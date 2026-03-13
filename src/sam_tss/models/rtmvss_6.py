"""
RTMVSS-6: SAM2-Based Real-Time Multispectral Video Semantic Segmentation Model

Architecture overview (uses SAM2 pretrained components):
  1. Dual-stream SAM2 ImageEncoder (Hiera trunk + FPN neck)
       - RGB and IR streams each load SAM2 pretrained weights
       - Four stages: 1/4, 1/8, 1/16, 1/32 resolution, all projected to 256-d by FPN
       - After scalp=1: three FPN levels at 1/4, 1/8, 1/16 (all 256-d)
  2. Cross-modal attention fusion (RGB ↔ IR) at each FPN level
  3. SAM2 MemoryEncoder: compact 64-d encoding of current fused features
  4. SAM2 MemoryAttention: cross-attention over stored temporal memory bank
  5. Segmentation head: FPN-style upsample → dense per-class prediction

Loading SAM2 pretrained weights:
    model.load_sam2_pretrained("path/to/sam2.1_hiera_t.pt")
    Variant configs:
        "hiera_tiny"      – embed_dim=96,  ~38 M params total
        "hiera_small"     – embed_dim=96,  ~46 M params total
        "hiera_base_plus" – embed_dim=112, ~80 M params total
        "hiera_large"     – embed_dim=144, ~224 M params total

Input/output interface is fully compatible with the existing MVNet training loop:
  forward(rgb_seq, ir_seq, step=0, epoch=0)
  → (output, aux_rgb, aux_thermal, aux_fusion, total_feas)
"""

import logging
import random
import warnings
from collections import deque
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# SAM2 modules (requires: pip install sam2)
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.memory_encoder import (
    CXBlock,
    Fuser,
    MaskDownSampler,
    MemoryEncoder,
)
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import RoPEAttention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM2 variant configurations
# ---------------------------------------------------------------------------

#: Maps variant name → kwargs for Hiera trunk and FPN backbone_channel_list.
SAM2_VARIANTS: Dict[str, Dict] = {
    "hiera_tiny": dict(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
        backbone_channel_list=[768, 384, 192, 96],
    ),
    "hiera_small": dict(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
        backbone_channel_list=[768, 384, 192, 96],
    ),
    "hiera_base_plus": dict(
        embed_dim=112,
        num_heads=2,
        stages=(2, 3, 16, 3),
        global_att_blocks=(12, 17, 21),
        window_pos_embed_bkg_spatial_size=(7, 7),
        backbone_channel_list=[896, 448, 224, 112],
    ),
    "hiera_large": dict(
        embed_dim=144,
        num_heads=2,
        stages=(2, 6, 36, 4),
        global_att_blocks=(23, 33, 43),
        window_pos_embed_bkg_spatial_size=(7, 7),
        backbone_channel_list=[1152, 576, 288, 144],
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 4-D tensors (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ---------------------------------------------------------------------------
# SAM2 ImageEncoder builder
# ---------------------------------------------------------------------------

def _build_sam2_image_encoder(variant: str) -> ImageEncoder:
    """Build a SAM2 ImageEncoder (trunk + FPN neck) for the given variant."""
    cfg = SAM2_VARIANTS[variant]
    trunk = Hiera(
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        stages=cfg["stages"],
        global_att_blocks=cfg["global_att_blocks"],
        window_pos_embed_bkg_spatial_size=cfg["window_pos_embed_bkg_spatial_size"],
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(num_pos_feats=256, normalize=True),
        d_model=256,
        backbone_channel_list=cfg["backbone_channel_list"],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    # scalp=1 drops the finest (1/4 stride-4) level from the FPN output, keeping
    # three levels at strides 8, 16, 32 (all projected to 256-d by the FPN neck).
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)


# ---------------------------------------------------------------------------
# SAM2 MemoryEncoder builder (no-mask variant)
# ---------------------------------------------------------------------------

def _build_sam2_memory_encoder() -> MemoryEncoder:
    """Build a SAM2 MemoryEncoder that encodes pixel features without mask conditioning."""
    return MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(num_pos_feats=64, normalize=True),
        mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
        fuser=Fuser(
            layer=CXBlock(dim=256, kernel_size=7, padding=3, use_dwconv=True),
            num_layers=2,
        ),
    )


# ---------------------------------------------------------------------------
# SAM2 MemoryAttention builder
# ---------------------------------------------------------------------------

def _build_sam2_memory_attention(feat_W: int, feat_H: int) -> MemoryAttention:
    """
    Build a SAM2 MemoryAttention module for feature maps of spatial size (feat_H, feat_W).

    ``feat_sizes=(feat_W, feat_H)`` must match the actual memory feature map to
    pre-compute RoPE frequencies correctly (avoids the broken square-map fallback
    inside RoPEAttention when feat_H != feat_W).
    """
    feat_sizes = (feat_W, feat_H)
    return MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=MemoryAttentionLayer(
            activation="relu",
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            d_model=256,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            self_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=feat_sizes,
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
            ),
            cross_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=feat_sizes,
                rope_k_repeat=True,
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64,
            ),
        ),
        num_layers=4,
        batch_first=True,  # SAM2 default; input/output remain seq-first (S,B,C)
    )


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

        self.proj_rgb = nn.Sequential(nn.Conv2d(inner, dim, 1), LayerNorm2d(dim))
        self.proj_ir = nn.Sequential(nn.Conv2d(inner, dim, 1), LayerNorm2d(dim))

        self.gate_rgb = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())
        self.gate_ir = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())

        self.norm_rgb = LayerNorm2d(dim)
        self.norm_ir = LayerNorm2d(dim)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, C, _, _ = q.shape
        nh, hd = self.num_heads, self.head_dim
        q = q.flatten(2).permute(0, 2, 1).view(B, H * W, nh, hd).permute(0, 2, 1, 3)
        k = k.flatten(2).permute(0, 2, 1).view(B, H * W, nh, hd).permute(0, 2, 1, 3)
        v = v.flatten(2).permute(0, 2, 1).view(B, H * W, nh, hd).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, H * W, C)
        return out.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, f_rgb: torch.Tensor, f_ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = f_rgb.shape
        q_r = self.q_rgb(self.norm_rgb(f_rgb))
        kv_i = self.kv_ir(self.norm_ir(f_ir))
        k_i, v_i = kv_i.chunk(2, dim=1)
        rgb_cross = self.proj_rgb(self._attn(q_r, k_i, v_i, H, W))
        gate_r = self.gate_rgb(torch.cat([f_rgb, rgb_cross], dim=1))
        f_rgb_out = f_rgb + gate_r * rgb_cross

        q_i = self.q_ir(self.norm_ir(f_ir))
        kv_r = self.kv_rgb(self.norm_rgb(f_rgb))
        k_r, v_r = kv_r.chunk(2, dim=1)
        ir_cross = self.proj_ir(self._attn(q_i, k_r, v_r, H, W))
        gate_i = self.gate_ir(torch.cat([f_ir, ir_cross], dim=1))
        f_ir_out = f_ir + gate_i * ir_cross

        return f_rgb_out, f_ir_out


# ---------------------------------------------------------------------------
# Modality Fusion (SE channel attention)
# ---------------------------------------------------------------------------

class ModalFuse(nn.Module):
    """Fuses RGB and IR features via SE channel-attention and 1×1 conv."""

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
# Segmentation Head
# ---------------------------------------------------------------------------

class SegHead(nn.Module):
    """Dense segmentation head: refine FPN features and upsample to full resolution."""

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
# Multi-scale decoder: fuses 3 SAM2 FPN levels → 1/4 res feature map
# ---------------------------------------------------------------------------

class FPNSegDecoder(nn.Module):
    """
    Top-down FPN decoder over the 3 SAM2 FPN levels (all 256-d at 1/4, 1/8, 1/16).
    Produces a single 256-d feature map at 1/4 resolution.
    """

    def __init__(self, fpn_dim: int = 256):
        super().__init__()
        # Refinement convs after top-down merge
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                LayerNorm2d(fpn_dim),
                nn.GELU(),
            )
            for _ in range(2)  # 2 merges: level2→level1, level1→level0
        ])

    def forward(self, fpn_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        fpn_feats: [level0 (1/4), level1 (1/8), level2 (1/16)] – all 256-d.
        Returns: (B, 256, H/4, W/4)
        """
        # Start from coarsest
        out = fpn_feats[2]
        out = F.interpolate(out, size=fpn_feats[1].shape[-2:], mode="nearest")
        out = fpn_feats[1] + out
        out = self.fuse[0](out)

        out = F.interpolate(out, size=fpn_feats[0].shape[-2:], mode="nearest")
        out = fpn_feats[0] + out
        out = self.fuse[1](out)
        return out


# ---------------------------------------------------------------------------
# Main Model: RTMVSS-6
# ---------------------------------------------------------------------------

class RTMVSS6(nn.Module):
    """
    RTMVSS-6: Real-Time Multispectral Video Semantic Segmentation using SAM2.

    Uses SAM2's pretrained ImageEncoder (Hiera + FPN), MemoryEncoder, and
    MemoryAttention modules.  Adds cross-modal attention (RGB ↔ IR) and a
    segmentation head on top.

    Temporal memory: each processed frame is encoded into 64-d memory tokens by
    SAM2's MemoryEncoder and stored in a FIFO deque (up to ``max_mem`` frames).
    On the first frame (empty bank) memory attention is skipped and the current
    features are used directly; from the second frame onwards SAM2's
    MemoryAttention enriches current features with stored past context.

    Args:
        num_classes:      Semantic classes (default 26 for MVSeg).
        sam2_variant:     One of "hiera_tiny", "hiera_small", "hiera_base_plus",
                          "hiera_large".  Selects the SAM2 backbone variant.
        img_size:         Spatial size (H, W) of input frames.  Used to compute
                          memory feature size (H/16, W/16) for correct RoPE init.
        max_mem:          Max stored memory frames (FIFO deque).
        share_encoder:    Share ImageEncoder weights between RGB and IR streams.
        memory_strategy:  "all" or "random" frame sampling.
        always_decode:    Produce output for every frame (not just the last).
        baseline_mode:    Disable temporal memory (ablation).
        stm_queue_size:   Alias for max_mem (MVNet compatibility).
        sample_rate:      Unused; kept for API compatibility.
    """

    def __init__(
        self,
        num_classes: int = 26,
        sam2_variant: str = "hiera_tiny",
        img_size: Tuple[int, int] = (320, 480),
        max_mem: int = 5,
        share_encoder: bool = False,
        memory_strategy: str = "all",
        always_decode: bool = False,
        baseline_mode: bool = False,
        # MVNet-compatibility kwargs
        stm_queue_size: int = 5,
        sample_rate: int = 1,
        # Legacy backbone kwargs (ignored when using SAM2 modules, kept for compat)
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        fpn_dim: int = 256,
        mem_dim: int = 64,
        drop_path_rate: float = 0.1,
        share_backbone: bool = False,
    ):
        super().__init__()
        if sam2_variant not in SAM2_VARIANTS:
            raise ValueError(f"sam2_variant must be one of {list(SAM2_VARIANTS)}; got {sam2_variant!r}")

        self.num_classes = num_classes
        self.sam2_variant = sam2_variant
        self.img_size = img_size
        self.max_mem = max(max_mem, stm_queue_size)
        self.memory_strategy = memory_strategy
        self.always_decode = always_decode
        self.baseline_mode = baseline_mode
        self.share_encoder = share_encoder or share_backbone

        # Memory feature map size: stride-16 of the input
        self._mem_H = img_size[0] // 16
        self._mem_W = img_size[1] // 16

        # ---- SAM2 Image Encoders (RGB + IR) ----
        self.encoder_rgb = _build_sam2_image_encoder(sam2_variant)
        if self.share_encoder:
            self.encoder_ir = self.encoder_rgb
        else:
            self.encoder_ir = _build_sam2_image_encoder(sam2_variant)

        # SAM2 FPN output: 3 levels at [H/4, H/8, H/16], all 256-d (after scalp=1)
        _fpn_dim = 256

        # ---- Cross-modal attention (RGB ↔ IR) at each FPN level ----
        self.cross_modal = nn.ModuleList([
            CrossModalAttention(_fpn_dim, num_heads=8)
            for _ in range(3)
        ])

        # ---- Modality fusion at each FPN level ----
        self.modal_fuse = nn.ModuleList([ModalFuse(_fpn_dim) for _ in range(3)])

        # ---- FPN decoders (one per modality stream) ----
        self.fpn_dec_rgb = FPNSegDecoder(_fpn_dim)
        self.fpn_dec_ir = FPNSegDecoder(_fpn_dim)
        self.fpn_dec_fused = FPNSegDecoder(_fpn_dim)

        # ---- SAM2 Memory Encoder (fused stream, no mask conditioning) ----
        self.memory_encoder = _build_sam2_memory_encoder()

        # ---- SAM2 Memory Attention ----
        self.memory_attention = _build_sam2_memory_attention(
            feat_W=self._mem_W, feat_H=self._mem_H
        )

        # Temporal positional encoding for memory slots (SAM2 style)
        self.maskmem_tpos_enc = nn.Parameter(
            torch.zeros(self.max_mem, 1, 1, 64)  # (max_mem, 1, 1, mem_dim)
        )
        nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        # Project memory-enriched features (256-d) back to 256-d for decoder
        self.mem_out_proj = nn.Sequential(
            nn.Conv2d(_fpn_dim, _fpn_dim, 1, bias=False),
            LayerNorm2d(_fpn_dim),
        )

        # ---- Segmentation heads ----
        self.seg_head_rgb = SegHead(_fpn_dim, num_classes)
        self.seg_head_ir = SegHead(_fpn_dim, num_classes)
        self.seg_head_fused = SegHead(_fpn_dim, num_classes)
        self.seg_head_final = nn.Conv2d(num_classes * 3, num_classes, 1)

        # ---- Memory bank (stateful) ----
        self._mem_bank: deque = deque(maxlen=self.max_mem)

        self.reset_hidden_state()
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init (only non-SAM2 modules need explicit init)
    # ------------------------------------------------------------------

    def _init_weights(self):
        for name, m in self.named_modules():
            # Skip SAM2 encoder modules – they are initialised by SAM2
            if name.startswith("encoder_rgb") or name.startswith("encoder_ir"):
                continue
            if name.startswith("memory_encoder") or name.startswith("memory_attention"):
                continue
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
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def load_sam2_pretrained(
        self,
        ckpt_path: str,
        load_encoder: bool = True,
        load_memory: bool = True,
        strict: bool = False,
    ) -> None:
        """
        Load SAM2 pretrained weights from a checkpoint file.

        Args:
            ckpt_path:    Path to the SAM2 .pt checkpoint.
            load_encoder: Load image_encoder weights into both RGB and IR encoders.
            load_memory:  Load memory_encoder and memory_attention weights.
            strict:       Whether to require all checkpoint keys to match.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)

        loaded, skipped = 0, 0

        def _load_submodule(module: nn.Module, prefix: str):
            nonlocal loaded, skipped
            sub_state = {
                k[len(prefix):]: v
                for k, v in state.items()
                if k.startswith(prefix)
            }
            result = module.load_state_dict(sub_state, strict=strict)
            loaded += len(sub_state)
            if result.missing_keys:
                logger.debug("Missing keys in %s: %s", prefix, result.missing_keys[:5])
            if result.unexpected_keys:
                skipped += len(result.unexpected_keys)

        if load_encoder:
            _load_submodule(self.encoder_rgb, "image_encoder.")
            if not self.share_encoder:
                _load_submodule(self.encoder_ir, "image_encoder.")
            logger.info("Loaded image_encoder weights from %s", ckpt_path)

        if load_memory:
            _load_submodule(self.memory_encoder, "memory_encoder.")
            _load_submodule(self.memory_attention, "memory_attention.")
            logger.info("Loaded memory_encoder + memory_attention weights from %s", ckpt_path)

        logger.info(
            "SAM2 pretrained load: %d tensors loaded, %d skipped from %s",
            loaded, skipped, ckpt_path,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_hidden_state(self):
        """Reset temporal memory bank (call at the start of each new clip)."""
        self._mem_bank.clear()

    # ------------------------------------------------------------------
    # Memory range selection
    # ------------------------------------------------------------------

    def _memory_range(self, seq_len: int) -> List[int]:
        if self.memory_strategy == "random":
            if seq_len <= 1:
                return list(range(seq_len))
            n_context = min(self.max_mem - 1, seq_len - 1)
            r = random.sample(range(seq_len - 1), n_context)
            r.append(seq_len - 1)
            return sorted(r)
        return list(range(seq_len))

    # ------------------------------------------------------------------
    # Per-frame encode + cross-modal fuse
    # ------------------------------------------------------------------

    def _encode_frame(
        self, rgb: torch.Tensor, ir: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Run SAM2 encoders on a single (B, 3, H, W) frame pair.
        Returns three lists of 3 FPN features (all 256-d): rgb, ir, fused.
        """
        enc_rgb = self.encoder_rgb(rgb)   # dict: backbone_fpn, vision_features, vision_pos_enc
        enc_ir = self.encoder_ir(ir)

        fpn_rgb = enc_rgb["backbone_fpn"]  # [f0(H/4), f1(H/8), f2(H/16)]
        fpn_ir = enc_ir["backbone_fpn"]

        # Cross-modal attention at each FPN level
        out_rgb, out_ir = list(fpn_rgb), list(fpn_ir)
        for i, cm in enumerate(self.cross_modal):
            out_rgb[i], out_ir[i] = cm(out_rgb[i], out_ir[i])

        # Modality fusion at each FPN level
        fused = [self.modal_fuse[i](out_rgb[i], out_ir[i]) for i in range(3)]

        return out_rgb, out_ir, fused

    # ------------------------------------------------------------------
    # Memory operations (SAM2-style)
    # ------------------------------------------------------------------

    def _apply_memory_attention(
        self,
        fused_fpn: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply SAM2 MemoryAttention to the coarsest (1/16) fused FPN level.
        Returns updated FPN list with enriched coarsest level.
        When the memory bank is empty (first frame), returns features unchanged.
        """
        if len(self._mem_bank) == 0:
            # No stored memory yet: skip memory attention on the first frame
            return fused_fpn

        B, C, H_m, W_m = fused_fpn[2].shape  # (B, 256, H/16, W/16)

        # Current frame features in seq-first format (S, B, C)
        curr = fused_fpn[2].flatten(2).permute(2, 0, 1)     # (H*W, B, 256)
        curr_pos = curr.new_zeros(H_m * W_m, B, 256)

        # Gather stored memory entries: each is (B, 64, Hm, Wm) + (B, 64, Hm, Wm)
        mem_list, pos_list = [], []
        bank_len = len(self._mem_bank)
        for slot_idx, entry in enumerate(self._mem_bank):
            feat = entry["vision_features"]   # (B, 64, Hm, Wm)
            pos = entry["vision_pos_enc"]      # (B, 64, Hm, Wm)
            # Flatten spatial dims → (Hm*Wm, B, 64) for seq-first format
            f_seq = feat.flatten(2).permute(2, 0, 1)
            p_seq = pos.flatten(2).permute(2, 0, 1)
            # Temporal positional encoding: slot 0 is the oldest entry in the deque,
            # so t_pos_idx 0 = oldest frame, bank_len-1 = most recently stored frame.
            t_pos_idx = min(bank_len - 1 - slot_idx, self.max_mem - 1)
            t_enc = self.maskmem_tpos_enc[t_pos_idx]  # (1, 1, 64)
            p_seq = p_seq + t_enc.view(1, 1, 64).expand_as(p_seq)
            mem_list.append(f_seq)
            pos_list.append(p_seq)
        memory = torch.cat(mem_list, dim=0)       # (T*Hm*Wm, B, 64)
        memory_pos = torch.cat(pos_list, dim=0)

        enriched = self.memory_attention(
            curr=curr,
            memory=memory,
            curr_pos=curr_pos,
            memory_pos=memory_pos,
        )  # (S, B, 256)
        # Reshape back to (B, 256, Hm, Wm) and residual-add
        enriched = enriched.permute(1, 2, 0).view(B, C, H_m, W_m)
        enriched = self.mem_out_proj(enriched)

        return fused_fpn[:2] + [fused_fpn[2] + enriched]


    def _memorise_frame(self, fused_fpn: List[torch.Tensor]) -> None:
        """
        Encode the fused bottleneck features and push to memory bank.
        Passes zero masks (skip_mask_sigmoid=True) so only pixel features are used.
        """
        pix_feat = fused_fpn[2]  # (B, 256, Hm, Wm)
        B, _, H_in_scale, W_in_scale = pix_feat.shape
        # MemoryEncoder expects masks at the original image resolution (H, W).
        # With skip_mask_sigmoid=True the mask is added after sigmoid, and passing
        # zeros is a no-op (skip_mask_sigmoid skips sigmoid; mask contribution = 0).
        mask_zeros = pix_feat.new_zeros(B, 1, self.img_size[0], self.img_size[1])
        enc = self.memory_encoder(pix_feat, mask_zeros, skip_mask_sigmoid=True)
        # SAM2's MemoryEncoder wraps vision_pos_enc in a list (PositionEmbeddingSine
        # returns a list for compatibility with multi-level use; we take the first element).
        pos_enc = enc["vision_pos_enc"]
        pos_enc = pos_enc[0] if isinstance(pos_enc, list) else pos_enc
        self._mem_bank.append({
            "vision_features": enc["vision_features"].detach(),
            "vision_pos_enc": pos_enc.detach(),
        })

    # ------------------------------------------------------------------
    # Per-frame decode
    # ------------------------------------------------------------------

    def _decode_frame(
        self,
        fpn_rgb: List[torch.Tensor],
        fpn_ir: List[torch.Tensor],
        fpn_fused: List[torch.Tensor],
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        f_rgb_dec = self.fpn_dec_rgb(fpn_rgb)
        f_ir_dec = self.fpn_dec_ir(fpn_ir)
        f_fused_dec = self.fpn_dec_fused(fpn_fused)

        p_r = self.seg_head_rgb(f_rgb_dec, H, W)
        p_i = self.seg_head_ir(f_ir_dec, H, W)
        p_f = self.seg_head_fused(f_fused_dec, H, W)
        pred = self.seg_head_final(torch.cat([p_r, p_i, p_f], dim=1))
        return pred, p_r, p_i, p_f

    # ------------------------------------------------------------------
    # Forward
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
        None,
    ]:
        """
        Args:
            rgb_seq:  (B, T, 3, H, W) RGB sequence.
            ir_seq:   (B, T, 3, H, W) IR / thermal sequence.
            step:     Global training step (unused; kept for API compat).
            epoch:    Training epoch (unused; kept for API compat).

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

            fpn_rgb, fpn_ir, fpn_fused = self._encode_frame(rgb_t, ir_t)

            if self.baseline_mode:
                if not (self.always_decode or t == T - 1):
                    continue
                pred, p_r, p_i, p_f = self._decode_frame(fpn_rgb, fpn_ir, fpn_fused, H, W)
                preds.append(pred)
                aux_rgb.append(p_r)
                aux_ir.append(p_i)
                aux_fused.append(p_f)
                continue

            # Decode the query frame with memory-enriched features
            if self.always_decode or t == T - 1:
                mem_fpn = self._apply_memory_attention(fpn_fused)
                pred, p_r, p_i, p_f = self._decode_frame(fpn_rgb, fpn_ir, mem_fpn, H, W)
                preds.append(pred)
                aux_rgb.append(p_r)
                aux_ir.append(p_i)
                aux_fused.append(p_f)

            # Memorise this frame (all frames, including the query)
            self._memorise_frame(fpn_fused)

        output_decoder = torch.stack(preds, dim=1)
        outputs_aux_rgb = torch.stack(aux_rgb, dim=1) if aux_rgb else None
        outputs_aux_thermal = torch.stack(aux_ir, dim=1) if aux_ir else None
        outputs_aux_fusion = torch.stack(aux_fused, dim=1) if aux_fused else None

        return output_decoder, outputs_aux_rgb, outputs_aux_thermal, outputs_aux_fusion, None


# ---------------------------------------------------------------------------
# Factory helper – build from MVNet-style args namespace
# ---------------------------------------------------------------------------

def build_rtmvss6_from_args(args) -> RTMVSS6:
    """
    Build RTMVSS6 from an argparse Namespace (MVNet-compatible).

    Recognised attributes (all optional, fall back to RTMVSS6 defaults):
        num_classes, sam2_variant, img_size, max_mem, share_encoder,
        memory_strategy, always_decode, baseline_mode, stm_queue_size,
        sample_rate.
    """
    kwargs = {}
    for key in (
        "num_classes", "sam2_variant", "img_size", "max_mem", "share_encoder",
        "memory_strategy", "always_decode", "baseline_mode",
        "stm_queue_size", "sample_rate",
    ):
        if hasattr(args, key):
            kwargs[key] = getattr(args, key)
    return RTMVSS6(**kwargs)
