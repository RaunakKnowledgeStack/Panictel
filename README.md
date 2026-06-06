# PanicIntel

PanicIntel is a multimodal crisis monitoring project that combines live news ingestion, vision-language scoring, retrieval memory, and sentiment analysis to produce a final crisis score.

## What it does

- Pulls live news from RSS feeds and the GDELT API
- Uses a T5 + ViT multimodal model to score each event
- Uses FAISS-based retrieval to compare new events with past ones
- Applies sentiment analysis as an extra signal
- Smooths the final score across similar events with graph propagation
- Exposes the system through a FastAPI backend and a browser dashboard

## Project Structure

- `app.py` - FastAPI application and API routes
- `config.py` - model and pipeline settings
- `model.py` - multimodal scoring model
- `pipeline.py` - news ingestion and scoring pipeline
- `rag.py` - retrieval memory using FAISS
- `train.py` - training loop and dataset code
- `static/index.html` - dashboard UI

## News Sources

The pipeline gets news from:

- BBC World RSS
- Reuters World RSS
- Google News RSS search for crisis/disaster topics
- GDELT Article API

## Requirements

You need Python 3.10+ and these main packages:

- `torch`
- `transformers`
- `fastapi`
- `uvicorn`
- `numpy`
- `requests`
- `feedparser`
- `Pillow`
- `vaderSentiment`
- `faiss`

## Setup

Install the dependencies:

```bash
pip install torch transformers fastapi uvicorn numpy requests feedparser pillow vaderSentiment faiss-cpu
```

If your system uses GPU FAISS or a different PyTorch build, install the matching version for your machine.

## Run the App

Start the backend:

```bash
uvicorn app:app --reload
```

Then open:

- `http://127.0.0.1:8000/`

## API Endpoints

- `GET /api/status` - shows whether models are loaded and how many events are available
- `GET /api/events` - returns the latest scored events
- `POST /api/run` - starts the news pipeline
- `POST /api/score` - scores a custom text input

Example request for `/api/score`:

```json
{
  "text": "There has been a major flood in the area."
}
```

## Training

The training script expects a dataset with:

- `text`
- `label`
- optional `img_url`

Example:

```python
from train import train
```

## Notes

- The project is designed to work with live online sources.
- If a news image fails to load, the pipeline falls back to a blank placeholder image.
- The model uses a frozen encoder setup to keep training and inference lighter.

## CV Summary

Suggested one-line description:

> Built a multimodal crisis monitoring system using ViT, T5, FAISS retrieval, sentiment analysis, and FastAPI.

