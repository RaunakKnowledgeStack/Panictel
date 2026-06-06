from __future__ import annotations

import io
import time

import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from config import CFG, DEVICE
from model import PanicIntel


def load_image(url: str | None, size=(224, 224)):
    try:
        if not url:
            raise ValueError("missing image url")
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB").resize(size)
    except Exception:
        return Image.new("RGB", size, (10, 10, 20))


class CrisisDataset(Dataset):
    """Expects a DataFrame with columns: text, label, img_url (optional)."""

    def __init__(self, df, tokenizer, vit_extractor, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.vit = vit_extractor
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        enc = self.tok(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        image = load_image(row.get("img_url"))
        pixels = self.vit(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixels,
            "label": label,
        }


def _optimizer(model, cfg):
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["LR"],
        weight_decay=1e-2,
    )


def train(model, train_df, tokenizer, vit_extractor, cfg=CFG, device=DEVICE):
    ds = CrisisDataset(train_df, tokenizer, vit_extractor, max_len=cfg["MAX_TEXT_LEN"])
    loader = DataLoader(
        ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    model = model.to(device)
    optimizer = _optimizer(model, cfg)
    steps = max(1, len(loader) * cfg["EPOCHS"])
    warmup = int(steps * cfg["WARMUP_RATIO"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, steps)
    loss_fn = nn.BCELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, cfg["EPOCHS"] + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, _ = model(ids, mask, pixels, return_logits=True)
                probs = torch.sigmoid(logits)
                loss = loss_fn(probs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch}/{cfg['EPOCHS']} - loss={avg_loss:.4f}")

    return model


def ablation(train_df, tokenizer, vit_extractor, cfg=CFG):
    """Compare partial freeze, full fine-tune, and aggressive freeze settings."""
    scenarios = [
        ("Partial", 4, False),
        ("Full fine-tune", -1, True),
        ("Aggressive", 10, False),
    ]

    for name, up_to, full in scenarios:
        c = dict(cfg, FREEZE_T5_UP_TO=up_to)
        model = PanicIntel(c).to(DEVICE)
        if full:
            for p in model.t5_enc.parameters():
                p.requires_grad = True
        t0 = time.time()
        train(model, train_df, tokenizer, vit_extractor, c, DEVICE)
        elapsed = time.time() - t0
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{name}: trainable={trainable/total:.1%} elapsed={elapsed:.1f}s")
