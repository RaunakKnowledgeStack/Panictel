from __future__ import annotations

import io
from datetime import datetime

import feedparser
import numpy as np
import requests
import torch
from PIL import Image
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import DEVICE


VADER = SentimentIntensityAnalyzer()

RSS_FEEDS = {
    "BBC": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Reuters": "https://feeds.reuters.com/reuters/worldNews",
    "GoogleNews": "https://news.google.com/rss/search?q=crisis+OR+disaster&hl=en-US&gl=US&ceid=US:en",
}


def _blank_image(size=(224, 224)):
    return Image.new("RGB", size, (10, 10, 20))


def fetch_rss(limit: int = 15):
    events = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                img_url = None
                if hasattr(entry, "media_content") and entry.media_content:
                    img_url = entry.media_content[0].get("url")
                events.append(
                    {
                        "source": source,
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                        "img_url": img_url,
                    }
                )
        except Exception as exc:
            print(f"[rss] {source}: {exc}")
    return events


def fetch_gdelt(limit: int = 20):
    try:
        response = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": "crisis OR disaster OR attack",
                "mode": "ArtList",
                "maxrecords": limit,
                "format": "json",
                "timespan": "1440",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            {
                "source": article.get("domain", "GDELT"),
                "title": article.get("title", ""),
                "summary": article.get("title", ""),
                "published": article.get("seendate", ""),
                "link": article.get("url", ""),
                "img_url": None,
            }
            for article in payload.get("articles", [])
        ]
    except Exception as exc:
        print(f"[gdelt] {exc}")
        return []


def ingest(limit: int = 20):
    raw = fetch_gdelt(limit) + fetch_rss(max(5, limit // 2))
    seen = set()
    out = []
    for event in raw:
        text = f"{event.get('title', '')} {event.get('summary', '')}".strip()
        if len(text) < 10 or text in seen:
            continue
        seen.add(text)
        event["text"] = text
        out.append(event)
    return out[:limit]


def load_image(url: str | None, size=(224, 224)):
    if not url:
        return _blank_image(size)
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB").resize(size)
    except Exception:
        return _blank_image(size)


def sentiment_score(text: str) -> float:
    compound = VADER.polarity_scores(text or "").get("compound", 0.0)
    return round((compound + 1.0) * 50.0, 2)


def cosine_sim_matrix(embs: np.ndarray) -> np.ndarray:
    arr = np.asarray(embs, dtype=np.float32)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / np.clip(norms, 1e-8, None)
    return normed @ normed.T


def sir_propagate(scores, embs, beta=0.30, theta=50.0, sim_cut=0.65, rounds=2):
    """Lightweight SIR-style smoothing over the similarity graph."""
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) <= 1:
        return scores

    sim = cosine_sim_matrix(embs)
    mask = (sim >= sim_cut).astype(np.float32)
    np.fill_diagonal(mask, 0.0)

    current = scores.copy()
    for _ in range(rounds):
        spread = np.maximum(0.0, current - theta)
        delta = beta * (mask @ spread)
        denom = np.maximum(mask.sum(axis=1), 1.0)
        current = np.clip(current + delta / denom, 0.0, 100.0)
    return current


def compute_gpi(text_s, rag_s, img_s, sent_s, w):
    return round(
        float(
            np.clip(
                w["text"] * text_s
                + w["rag"] * rag_s
                + w["image"] * img_s
                + w["sentiment"] * sent_s,
                0,
                100,
            )
        ),
        2,
    )


@torch.inference_mode()
def run_pipeline(model, tokenizer, vit_extractor, rag, cfg, limit: int = 20):
    model.eval()
    events = ingest(limit)
    if not events:
        return []

    weights = cfg["GPI_WEIGHTS"]
    results = []
    embs = []

    for event in events:
        text = event.get("text", "")
        image = load_image(event.get("img_url"), size=cfg.get("IMAGE_SIZE", (224, 224)))

        enc = tokenizer(
            text,
            max_length=cfg["MAX_TEXT_LEN"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        pixels = vit_extractor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)

        text_score, hidden = model(enc["input_ids"], enc["attention_mask"], pixels)
        text_score = float(text_score.item() * 100.0)
        embedding = hidden.squeeze(0).cpu().numpy()

        rag_score = rag.rag_score(embedding)
        sent_score = sentiment_score(text)

        blank = tokenizer(
            cfg.get("IMAGE_ONLY_PROMPT", "crisis"),
            max_length=14,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        image_score, _ = model(blank["input_ids"], blank["attention_mask"], pixels)
        image_score = float(image_score.item() * 100.0)

        gpi = compute_gpi(text_score, rag_score, image_score, sent_score, weights)

        results.append(
            {
                **event,
                "text_score": round(text_score, 2),
                "rag_score": round(float(rag_score), 2),
                "img_score": round(image_score, 2),
                "sent_score": round(sent_score, 2),
                "gpi": gpi,
                "_emb": embedding,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        embs.append(embedding)

        rag.add(
            embedding,
            {
                "gpi": gpi,
                "text": text,
                "ts": datetime.utcnow().isoformat(),
            },
        )

    final_scores = sir_propagate(
        [item["gpi"] for item in results],
        np.stack(embs),
        beta=cfg.get("SIR_BETA", 0.3),
        theta=cfg.get("SIR_THETA", 50.0),
        sim_cut=cfg.get("SIR_SIM_CUTOFF", 0.65),
        rounds=cfg.get("SIR_ROUNDS", 2),
    )

    for item, final in zip(results, final_scores):
        item["gpi_final"] = round(float(final), 2)
        item.pop("_emb", None)

    results.sort(key=lambda row: row["gpi_final"], reverse=True)
    return results
