from __future__ import annotations

from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CFG = {
    "VIT_MODEL": "google/vit-base-patch16-224-in21k",
    "T5_MODEL": "t5-base",
    "MAX_TEXT_LEN": 128,
    "BATCH_SIZE": 8,
    "LR": 2e-4,
    "EPOCHS": 4,
    "WARMUP_RATIO": 0.1,
    "FREEZE_T5_UP_TO": 4,
    "N_VISUAL_TOKENS": 8,
    "HIDDEN_SIZE": 256,
    "TOP_K": 30,
    "RSS_LIMIT": 15,
    "GDELT_LIMIT": 20,
    "IMAGE_SIZE": (224, 224),
    "IMAGE_ONLY_PROMPT": "crisis",
    "GPI_WEIGHTS": {
        "text": 0.35,
        "rag": 0.30,
        "image": 0.20,
        "sentiment": 0.15,
    },
    "SIR_BETA": 0.30,
    "SIR_THETA": 50.0,
    "SIR_SIM_CUTOFF": 0.65,
    "SIR_ROUNDS": 2,
}
