from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import T5Tokenizer, ViTFeatureExtractor

from config import BASE_DIR, CFG, DEVICE
from model import PanicIntel
from pipeline import run_pipeline, sentiment_score
from rag import CrisisRAG


app = FastAPI(title="PanicIntel API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



state = {
    "model": None,
    "tokenizer": None,
    "vit_extractor": None,
    "rag": None,
    "events": [],
    "loading": True,
    "last_run": None,
}


def _load_models():
    tokenizer = T5Tokenizer.from_pretrained(CFG["T5_MODEL"])
    vit_extractor = ViTFeatureExtractor.from_pretrained(CFG["VIT_MODEL"])
    model = PanicIntel(CFG).to(DEVICE).eval()
    rag = CrisisRAG(dim=int(model.t5_enc.config.d_model), top_k=CFG["TOP_K"])

    rag_path = BASE_DIR / "rag_index"
    rag_faiss = BASE_DIR / "rag_index.faiss"
    rag_meta = BASE_DIR / "rag_index.meta.json"
    if rag_faiss.exists() and rag_meta.exists():
        try:
            rag.load(str(rag_path))
        except Exception as exc:
            print(f"[rag] load skipped: {exc}")

    state.update(
        {
            "model": model,
            "tokenizer": tokenizer,
            "vit_extractor": vit_extractor,
            "rag": rag,
            "loading": False,
        }
    )
    print("[PanicIntel] models loaded")


@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_models)





@app.get("/api/status")
def status():
    return {
        "loading": state["loading"],
        "device": str(DEVICE),
        "event_count": len(state["events"]),
        "last_run": state["last_run"],
    }


async def _run():
    results = run_pipeline(
        state["model"],
        state["tokenizer"],
        state["vit_extractor"],
        state["rag"],
        CFG,
        limit=CFG["RSS_LIMIT"],
    )
    state["events"] = results
    state["last_run"] = datetime.utcnow().isoformat()


@app.post("/api/run")
async def trigger_run(background_tasks: BackgroundTasks):
    if state["loading"]:
        return {"error": "Models are still loading"}
    background_tasks.add_task(_run)
    return {"message": "Pipeline started"}


@app.get("/api/events")
def get_events():
    return {"events": state["events"], "last_run": state["last_run"]}


@app.post("/api/score")
async def score_text(payload: dict):
    if state["loading"]:
        return {"error": "Models are still loading"}

    text = str(payload.get("text", "")).strip()
    if not text:
        return {"error": "No text provided"}

    model = state["model"]
    tokenizer = state["tokenizer"]
    vit_extractor = state["vit_extractor"]

    enc = tokenizer(
        text,
        max_length=CFG["MAX_TEXT_LEN"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    image = Image.new("RGB", CFG["IMAGE_SIZE"], (10, 10, 20))
    pixels = vit_extractor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        score, hidden = model(enc["input_ids"], enc["attention_mask"], pixels)

    embedding = hidden.squeeze(0).cpu().numpy()
    rag_score = state["rag"].rag_score(embedding)
    sent_score = sentiment_score(text)
    text_score = round(float(score.item() * 100.0), 2)
    image_score = 50.0
    gpi = round(
        float(
            np.clip(
                CFG["GPI_WEIGHTS"]["text"] * text_score
                + CFG["GPI_WEIGHTS"]["rag"] * rag_score
                + CFG["GPI_WEIGHTS"]["image"] * image_score
                + CFG["GPI_WEIGHTS"]["sentiment"] * sent_score,
                0,
                100,
            )
        ),
        2,
    )

    return {
        "text": text,
        "gpi": gpi,
        "text_score": text_score,
        "rag_score": round(float(rag_score), 2),
        "img_score": image_score,
        "sent_score": sent_score,
    }
