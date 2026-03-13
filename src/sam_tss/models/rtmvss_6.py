"""
RTMVSS-6: SAM2-Based Real-Time Multispectral Video Semantic Segmentation Model

Architecture (inspired by SAM-DAQ / rdvsod.py, AAAI 2026):
  1. build_sam2_video_predictor → self.sam2 (full SAM2 with pretrained weights)
  2. Dual-stream encoding via self.sam2.image_encoder.trunk (shared Hiera backbone)
       - BiModalStageAdapter residuals injected at each of the 4 trunk stage transitions
         for cross-modal RGB ↔ IR feature alignment
       - FPN neck: self.sam2.image_encoder.neck → 3 output levels (256-d) at strides 4, 8, 16
  3. FPN-level modality fusion (ModalFuse) → RGB, IR and fused feature pyramids
  4. SAM2 MemoryEncoder (self.sam2.memory_encoder) → 64-d memory tokens
  5. SAM2 MemoryAttention (self.sam2.memory_attention) → temporal context via RoPE cross-attention
  6. FPN decoder + per-modality segmentation heads → dense 26-class output

Building the model:
    from sam2.build_sam import build_sam2_video_predictor
    model = RTMVSS6(
        sam2_config="sam2.1/sam2.1_hiera_t.yaml",
        sam2_ckpt="sam2.1_hiera_tiny.pt",
        num_classes=26,
    )

Input/output interface is compatible with the existing MVNet training loop:
    forward(rgb_seq, ir_seq, step=0, epoch=0)
    → (output, aux_rgb, aux_thermal, aux_fusion, total_feas)
"""

import logging
import random
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# SAM2 – build full video predictor with pretrained weights
from sam2.build_sam import build_sam2_video_predictor

logger = logging.getLogger(__name__)


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
# Bi-modal stage adapter (following rdvsod.py / SAM-DAQ)
# ---------------------------------------------------------------------------

class BiModalStageAdapter(nn.Module):
    """
    Lightweight bi-modal cross-modal adapter.

    Takes same-stage RGB and IR features (each of size ``in_channels``) and
    returns two residual tensors of size ``out_channels`` that are added back to
    the next stage's RGB and IR features respectively.  Spatial downsampling is
    handled by the caller (F.interpolate) so this module is purely channel-wise.

    Follows the ``bi_modal_parallel_adapter`` design from rdvsod.py.
    """

    def __init__(self, in_channels: int, out_channels: int, bottleneck: int = 64):
        super().__init__()
        self.up = nn.Conv2d(in_channels * 2, bottleneck, 3, padding=1, bias=True)
        self.act = nn.GELU()
        self.down = nn.Conv2d(bottleneck, out_channels * 2, 1)
        # Skip connection to keep gradients flowing when in_channels != out_channels
        self.skip = nn.Conv2d(in_channels * 2, out_channels * 2, 1)
        self.out_channels = out_channels

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([rgb, ir], dim=1)
        out = self.down(self.act(self.up(x))) + self.skip(x)
        rgb_out, ir_out = out.chunk(2, dim=1)
        return rgb_out, ir_out


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
# Multi-scale FPN decoder: fuses 3 SAM2 FPN levels → 1/4 res feature map
# ---------------------------------------------------------------------------

class FPNSegDecoder(nn.Module):
    """
    Top-down FPN decoder over the 3 SAM2 FPN levels (all 256-d at strides 4, 8, 16).
    Produces a single 256-d feature map at stride-4 (1/4) resolution.
    """

    def __init__(self, fpn_dim: int = 256):
        super().__init__()
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                LayerNorm2d(fpn_dim),
                nn.GELU(),
            )
            for _ in range(2)
        ])

    def forward(self, fpn_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        fpn_feats: [level0 (stride-4), level1 (stride-8), level2 (stride-16)] – all 256-d.
        Returns: (B, 256, H/4, W/4)
        """
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

    Builds the full SAM2VideoPredictor via ``build_sam2_video_predictor`` so all
    pretrained weights (backbone, memory encoder, memory attention) are loaded at
    construction time.  Cross-modal BiModalStageAdapter residuals align RGB and IR
    features at each Hiera trunk stage; a semantic segmentation head replaces SAM2's
    mask decoder.

    Args:
        sam2_config:      SAM2 config file (relative to sam2/configs), e.g.
                          ``"sam2.1/sam2.1_hiera_t.yaml"``.
        sam2_ckpt:        Path to SAM2 checkpoint .pt file, or None (random init).
        num_classes:      Semantic classes (default 26 for MVSeg).
        img_size:         Spatial size (H, W) of input frames.
        max_mem:          Maximum number of stored memory frames (FIFO deque).
        memory_strategy:  ``"all"`` or ``"random"`` frame sampling.
        always_decode:    Produce output logits for every frame, not just the last.
        baseline_mode:    Disable temporal memory (ablation).
        stm_queue_size:   Alias for max_mem (MVNet compatibility).
        sample_rate:      Unused; kept for API compatibility.
    """

    def __init__(
        self,
        sam2_config: str = "sam2.1/sam2.1_hiera_t.yaml",
        sam2_ckpt: Optional[str] = None,
        num_classes: int = 26,
        img_size: Tuple[int, int] = (320, 480),
        max_mem: int = 5,
        memory_strategy: str = "all",
        always_decode: bool = False,
        baseline_mode: bool = False,
        # MVNet-compatibility kwargs
        stm_queue_size: int = 5,
        sample_rate: int = 1,
        # Legacy kwargs (ignored; kept for API compatibility with build_rtmvss6_from_args)
        sam2_variant: str = "hiera_tiny",
        share_encoder: bool = False,
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

        self.num_classes = num_classes
        self.img_size = img_size
        self.max_mem = max(max_mem, stm_queue_size)
        self.memory_strategy = memory_strategy
        self.always_decode = always_decode
        self.baseline_mode = baseline_mode

        # Memory feature map size: stride-16 of the input
        self._mem_H = img_size[0] // 16
        self._mem_W = img_size[1] // 16

        # ---- Build full SAM2VideoPredictor (loads pretrained weights if ckpt given) ----
        # Initialise on CPU; the caller (trainer) moves the model to the target device.
        self.sam2 = build_sam2_video_predictor(
            config_file=sam2_config,
            ckpt_path=sam2_ckpt,
            device="cpu",
            mode="train",
        )

        # Trunk channel list: coarse-to-fine, e.g. [768, 384, 192, 96] for hiera_tiny
        # Reverse to get fine-to-coarse: [96, 192, 384, 768]
        ch = list(reversed(self.sam2.image_encoder.trunk.channel_list))

        # ---- Bi-modal stage adapters (following rdvsod.py) ----
        # adapter_i: takes stage-(i-1) [rgb, ir] features → residuals for stage-i
        # Spatial sizes are aligned by F.interpolate in _encode_frame.
        self.stage_adapter1 = BiModalStageAdapter(ch[0], ch[1])
        self.stage_adapter2 = BiModalStageAdapter(ch[1], ch[2])
        self.stage_adapter3 = BiModalStageAdapter(ch[2], ch[3])
        # Final refinement: refine stage-3 RGB features using IR
        self.stage_adapter4 = BiModalStageAdapter(ch[3], ch[3])

        # ---- FPN-level modality fusion ----
        _fpn_dim = self.sam2.image_encoder.neck.d_model  # 256
        self.modal_fuse = nn.ModuleList([ModalFuse(_fpn_dim) for _ in range(3)])

        # ---- FPN decoders (one per modality stream) ----
        self.fpn_dec_rgb = FPNSegDecoder(_fpn_dim)
        self.fpn_dec_ir = FPNSegDecoder(_fpn_dim)
        self.fpn_dec_fused = FPNSegDecoder(_fpn_dim)

        # Project memory-enriched features (256-d) for decoder
        self.mem_out_proj = nn.Sequential(
            nn.Conv2d(_fpn_dim, _fpn_dim, 1, bias=False),
            LayerNorm2d(_fpn_dim),
        )

        # ---- Segmentation heads ----
        self.seg_head_rgb = SegHead(_fpn_dim, num_classes)
        self.seg_head_ir = SegHead(_fpn_dim, num_classes)
        self.seg_head_fused = SegHead(_fpn_dim, num_classes)
        self.seg_head_final = nn.Conv2d(num_classes * 3, num_classes, 1)

        # ---- Memory bank (stateful FIFO) ----
        self._mem_bank: deque = deque(maxlen=self.max_mem)

        self.reset_hidden_state()
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init (only non-SAM2 modules need explicit init)
    # ------------------------------------------------------------------

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith("sam2"):
                continue  # SAM2 weights initialised by build_sam2_video_predictor
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
    # Per-frame dual-stream encoding with stage adapters
    # ------------------------------------------------------------------

    def _encode_frame(
        self, rgb: torch.Tensor, ir: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode a (B, 3, H, W) RGB+IR frame pair through the shared SAM2 backbone.

        Uses the shared ``self.sam2.image_encoder`` trunk for both modalities and
        injects BiModalStageAdapter residuals at each stage transition, following
        the design of rdvsod.py (SAM-DAQ, AAAI 2026).

        Returns three lists of 3 FPN features (each 256-d):
            fpn_rgb, fpn_ir, fpn_fused
        """
        # ---- Run trunk for both modalities (all 4 stages at once) ----
        trunk_rgb = list(self.sam2.image_encoder.trunk(rgb))
        trunk_ir = list(self.sam2.image_encoder.trunk(ir))
        # trunk outputs: [stage0 (finest), stage1, stage2, stage3 (coarsest)]
        # Sizes: stage0=(B,C0,H/4,W/4), stage1=(B,C1,H/8,W/8), ...

        # ---- Inject cross-modal adapter residuals ----
        # adapter1: stage0 features → residuals interpolated to stage1 size
        skip_rgb, skip_ir = self.stage_adapter1(trunk_rgb[0], trunk_ir[0])
        skip_rgb = F.interpolate(skip_rgb, size=trunk_rgb[1].shape[-2:],
                                 mode="bilinear", align_corners=False)
        skip_ir = F.interpolate(skip_ir, size=trunk_ir[1].shape[-2:],
                                mode="bilinear", align_corners=False)
        trunk_rgb[1] = trunk_rgb[1] + skip_rgb
        trunk_ir[1] = trunk_ir[1] + skip_ir

        # adapter2: stage1 features → residuals interpolated to stage2 size
        skip_rgb, skip_ir = self.stage_adapter2(trunk_rgb[1], trunk_ir[1])
        skip_rgb = F.interpolate(skip_rgb, size=trunk_rgb[2].shape[-2:],
                                 mode="bilinear", align_corners=False)
        skip_ir = F.interpolate(skip_ir, size=trunk_ir[2].shape[-2:],
                                mode="bilinear", align_corners=False)
        trunk_rgb[2] = trunk_rgb[2] + skip_rgb
        trunk_ir[2] = trunk_ir[2] + skip_ir

        # adapter3: stage2 features → residuals interpolated to stage3 size
        skip_rgb, skip_ir = self.stage_adapter3(trunk_rgb[2], trunk_ir[2])
        skip_rgb = F.interpolate(skip_rgb, size=trunk_rgb[3].shape[-2:],
                                 mode="bilinear", align_corners=False)
        skip_ir = F.interpolate(skip_ir, size=trunk_ir[3].shape[-2:],
                                mode="bilinear", align_corners=False)
        trunk_rgb[3] = trunk_rgb[3] + skip_rgb
        trunk_ir[3] = trunk_ir[3] + skip_ir

        # adapter4: final stage3 refinement of RGB using IR context only.
        # The IR stream already received RGB context from adapters 1-3, so we only
        # compute the RGB residual here and discard the unused IR output.
        skip_rgb, _ = self.stage_adapter4(trunk_rgb[3], trunk_ir[3])
        trunk_rgb[3] = trunk_rgb[3] + skip_rgb

        # ---- FPN neck (shared weights) for both streams ----
        fpn_rgb, _ = self.sam2.image_encoder.neck(trunk_rgb)
        fpn_ir, _ = self.sam2.image_encoder.neck(trunk_ir)

        # Apply scalp (drop lowest-resolution level)
        scalp = self.sam2.image_encoder.scalp
        if scalp > 0:
            fpn_rgb = fpn_rgb[:-scalp]
            fpn_ir = fpn_ir[:-scalp]
        # fpn_rgb/fpn_ir: [level0 (stride-4), level1 (stride-8), level2 (stride-16)]

        # ---- FPN-level modality fusion ----
        fused = [self.modal_fuse[i](fpn_rgb[i], fpn_ir[i]) for i in range(3)]

        return fpn_rgb, fpn_ir, fused

    # ------------------------------------------------------------------
    # Memory operations (SAM2-style)
    # ------------------------------------------------------------------

    def _apply_memory_attention(
        self,
        fused_fpn: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply SAM2 MemoryAttention to the coarsest (stride-16) fused FPN level.
        When the memory bank is empty (first frame), returns features unchanged.
        """
        if len(self._mem_bank) == 0:
            return fused_fpn

        B, C, H_m, W_m = fused_fpn[2].shape

        # Current frame features: seq-first (S, B, C)
        curr = fused_fpn[2].flatten(2).permute(2, 0, 1)
        curr_pos = curr.new_zeros(H_m * W_m, B, C)

        # Gather stored memory frames
        mem_list, pos_list = [], []
        bank_len = len(self._mem_bank)
        # SAM2's maskmem_tpos_enc has shape (num_maskmem, 1, 1, mem_dim=64)
        num_maskmem = self.sam2.maskmem_tpos_enc.shape[0]
        for slot_idx, entry in enumerate(self._mem_bank):
            feat = entry["vision_features"]   # (B, 64, Hm, Wm)
            pos = entry["vision_pos_enc"]      # (B, 64, Hm, Wm)
            f_seq = feat.flatten(2).permute(2, 0, 1)  # (Hm*Wm, B, 64)
            p_seq = pos.flatten(2).permute(2, 0, 1)
            # t_pos_idx: 0=oldest, bank_len-1=newest → most-recent gets index 0 of tpos_enc
            t_pos_idx = min(bank_len - 1 - slot_idx, num_maskmem - 1)
            t_enc = self.sam2.maskmem_tpos_enc[t_pos_idx]  # (1, 1, 64)
            p_seq = p_seq + t_enc.view(1, 1, 64).expand_as(p_seq)
            mem_list.append(f_seq)
            pos_list.append(p_seq)

        memory = torch.cat(mem_list, dim=0)       # (T*Hm*Wm, B, 64)
        memory_pos = torch.cat(pos_list, dim=0)

        enriched = self.sam2.memory_attention(
            curr=curr,
            memory=memory,
            curr_pos=curr_pos,
            memory_pos=memory_pos,
        )  # (S, B, C)
        enriched = enriched.permute(1, 2, 0).view(B, C, H_m, W_m)
        enriched = self.mem_out_proj(enriched)

        return fused_fpn[:2] + [fused_fpn[2] + enriched]

    def _memorise_frame(self, fused_fpn: List[torch.Tensor]) -> None:
        """
        Encode the coarsest fused FPN features into 64-d memory tokens and push to bank.
        Uses self.sam2.memory_encoder (SAM2's pretrained module).
        """
        pix_feat = fused_fpn[2]  # (B, 256, Hm, Wm)
        B = pix_feat.shape[0]
        # MemoryEncoder expects a mask at original image resolution; zero mask = no mask conditioning
        mask_zeros = pix_feat.new_zeros(B, 1, self.img_size[0], self.img_size[1])
        enc = self.sam2.memory_encoder(pix_feat, mask_zeros, skip_mask_sigmoid=True)
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
            total_feas:           None
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

            # Apply memory attention and decode the query frame
            if self.always_decode or t == T - 1:
                mem_fpn = self._apply_memory_attention(fpn_fused)
                pred, p_r, p_i, p_f = self._decode_frame(fpn_rgb, fpn_ir, mem_fpn, H, W)
                preds.append(pred)
                aux_rgb.append(p_r)
                aux_ir.append(p_i)
                aux_fused.append(p_f)

            # Memorise this frame
            self._memorise_frame(fpn_fused)

        output_decoder = torch.stack(preds, dim=1)
        outputs_aux_rgb = torch.stack(aux_rgb, dim=1) if aux_rgb else None
        outputs_aux_thermal = torch.stack(aux_ir, dim=1) if aux_ir else None
        outputs_aux_fusion = torch.stack(aux_fused, dim=1) if aux_fused else None

        return output_decoder, outputs_aux_rgb, outputs_aux_thermal, outputs_aux_fusion, None


# ---------------------------------------------------------------------------
# Factory helper – build from MVNet-style args namespace
# ---------------------------------------------------------------------------

#: Maps the legacy ``sam2_variant`` string to the SAM2 config file name.
_VARIANT_TO_CONFIG = {
    "hiera_tiny":      "sam2.1/sam2.1_hiera_t.yaml",
    "hiera_small":     "sam2.1/sam2.1_hiera_s.yaml",
    "hiera_base_plus": "sam2.1/sam2.1_hiera_b+.yaml",
    "hiera_large":     "sam2.1/sam2.1_hiera_l.yaml",
}


def build_rtmvss6_from_args(args) -> RTMVSS6:
    """
    Build RTMVSS6 from an argparse Namespace (MVNet-compatible).

    Recognised attributes (all optional, fall back to RTMVSS6 defaults):
        sam2_config, sam2_ckpt, sam2_variant (legacy alias for sam2_config),
        num_classes, img_size, max_mem, memory_strategy, always_decode,
        baseline_mode, stm_queue_size, sample_rate.
    """
    kwargs = {}

    # Resolve SAM2 config: prefer explicit sam2_config, fall back to sam2_variant mapping
    if hasattr(args, "sam2_config"):
        kwargs["sam2_config"] = args.sam2_config
    elif hasattr(args, "sam2_variant"):
        variant = args.sam2_variant
        kwargs["sam2_config"] = _VARIANT_TO_CONFIG.get(
            variant, "sam2.1/sam2.1_hiera_t.yaml"
        )

    if hasattr(args, "sam2_ckpt"):
        kwargs["sam2_ckpt"] = args.sam2_ckpt

    for key in (
        "num_classes", "img_size", "max_mem", "memory_strategy",
        "always_decode", "baseline_mode", "stm_queue_size", "sample_rate",
    ):
        if hasattr(args, key):
            kwargs[key] = getattr(args, key)

    return RTMVSS6(**kwargs)
