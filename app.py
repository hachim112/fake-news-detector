from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
from googlesearch import search

# -------------------
# FASTAPI APP SETUP
# -------------------
app = FastAPI(title="Fake News Detector", version="1.1")

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates (HTML frontend)
templates = Jinja2Templates(directory="templates")

# -------------------
# INPUT MODEL
# -------------------
class NewsItem(BaseModel):
    text: str

# -------------------
# LOAD MODEL LOCALLY (No token needed)
# -------------------
classifier = pipeline("text-classification", model="Falah/News_Detection")

# -------------------
# LABEL MAPPING
# -------------------
LABEL_MAP = {
    "LABEL_0": "REAL",
    "LABEL_1": "FAKE",
    "REAL": "REAL",
    "FAKE": "FAKE"
}

# -------------------
# HELPER: GET SOURCES
# -------------------
def get_sources(query, num_results=3):
    """Return a list of URLs for the query."""
    urls = []
    try:
        for url in search(query, num_results=num_results, lang="en"):
            urls.append(url)
    except Exception as e:
        urls.append(f"Error fetching URLs: {str(e)}")
    return urls

# -------------------
# ROUTES
# -------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect_fake_news(item: NewsItem):
    """Detect if text is FAKE or REAL with URLs."""
    try:
        # Make prediction
        result = classifier(item.text)

        # Extract top prediction
        best = max(result, key=lambda x: x['score'])
        raw_label = best["label"]
        label = LABEL_MAP.get(raw_label, raw_label)
        confidence = round(best["score"] * 100, 2)

        # Explanation logic
        explanation = (
            f"The model predicts this text is '{label}' with {confidence}% confidence. "
            + ("⚠️ Sensationalist language detected — likely fake news."
               if label == "FAKE" else "✅ Resembles trustworthy news.")
        )

        # Get URLs to verify/fact-check
        query_type = "debunk" if label == "FAKE" else "news"
        urls = get_sources(f"{query_type} {item.text}")

        return {
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "sources": urls
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
