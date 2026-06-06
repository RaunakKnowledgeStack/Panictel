from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import T5EncoderModel, ViTModel


@dataclass(frozen=True)
class ModelReport:
    total: int
    trainable: int

    @property
    def frozen(self) -> int:
        return self.total - self.trainable


class VisualTokenProjector(nn.Module):
    """Project a ViT CLS embedding into a learned sequence of T5 prefix tokens."""

    def __init__(self, vit_dim: int, t5_dim: int, n_tokens: int, dropout: float = 0.10):
        super().__init__()
        self.n_tokens = n_tokens
        self.t5_dim = t5_dim
        self.net = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, t5_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(t5_dim * 2, n_tokens * t5_dim),
        )
        self.out_norm = nn.LayerNorm(t5_dim)

    def forward(self, vit_cls: torch.Tensor) -> torch.Tensor:
        tokens = self.net(vit_cls).view(vit_cls.size(0), self.n_tokens, self.t5_dim)
        return self.out_norm(tokens)


class PanicIntel(nn.Module):
    """Multimodal crisis scorer with frozen encoders and a trainable fusion head."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        self.vit = ViTModel.from_pretrained(cfg["VIT_MODEL"])
        for param in self.vit.parameters():
            param.requires_grad = False

        self.t5_enc = T5EncoderModel.from_pretrained(cfg["T5_MODEL"])
        self.t5_embed = self.t5_enc.get_input_embeddings()
        self._freeze_t5(cfg.get("FREEZE_T5_UP_TO", 0))

        vit_dim = int(self.vit.config.hidden_size)
        t5_dim = int(self.t5_enc.config.d_model)
        n_tokens = int(cfg.get("N_VISUAL_TOKENS", 8))

        self.vis_proj = VisualTokenProjector(vit_dim=vit_dim, t5_dim=t5_dim, n_tokens=n_tokens)
        self.fuse_norm = nn.LayerNorm(t5_dim)
        self.dropout = nn.Dropout(0.10)
        hidden_size = int(cfg.get("HIDDEN_SIZE", 256))
        self.scorer = nn.Sequential(
            nn.Linear(t5_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def _freeze_t5(self, up_to: int) -> None:
        for param in self.t5_embed.parameters():
            param.requires_grad = False

        for idx, block in enumerate(self.t5_enc.encoder.block):
            freeze_block = up_to >= 0 and idx <= up_to
            for param in block.parameters():
                param.requires_grad = not freeze_block

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_logits: bool = False,
    ):
        batch_size = input_ids.size(0)

        text_emb = self.t5_embed(input_ids)
        vit_out = self.vit(pixel_values=pixel_values)
        visual_tokens = self.vis_proj(vit_out.last_hidden_state[:, 0])

        fused_emb = torch.cat([visual_tokens, text_emb], dim=1)
        visual_mask = attention_mask.new_ones((batch_size, visual_tokens.size(1)))
        full_mask = torch.cat([visual_mask, attention_mask], dim=1)

        enc_out = self.t5_enc(inputs_embeds=fused_emb, attention_mask=full_mask)
        hidden = enc_out.last_hidden_state

        mask = full_mask.unsqueeze(-1).type_as(hidden)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(self.fuse_norm(pooled))

        logits = self.scorer(pooled).squeeze(-1)
        score = torch.sigmoid(logits)
        if return_logits:
            return logits, pooled
        return score, pooled

    def parameter_report(self) -> ModelReport:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total   : {total:,}")
        print(f"Trainable: {trainable:,}")
        print(f"Frozen  : {total - trainable:,}")
        return ModelReport(total=total, trainable=trainable)
