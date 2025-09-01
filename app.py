from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline

# FastAPI setup
app = FastAPI(title="Fake News Detector")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Input model
class NewsItem(BaseModel):
    text: str

# Load a small text classification model locally
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Label mapping
LABEL_MAP = {
    "POSITIVE": "REAL",
    "NEGATIVE": "FAKE"
}

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect(item: NewsItem):
    try:
        # Make prediction
        result = classifier(item.text)
        best = max(result, key=lambda x: x["score"])
        label = LABEL_MAP.get(best["label"], best["label"])
        confidence = round(best["score"] * 100, 2)

        explanation = (
            f"The model predicts this text is '{label}' with {confidence}% confidence. "
            + ("⚠️ Likely fake news." if label == "FAKE" else "✅ Resembles trustworthy news.")
        )

        # Manual top sources (for now)
        sources = ["https://www.bbc.com", "https://www.cnn.com", "https://www.reuters.com"]

        return {
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "sources": sources
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
