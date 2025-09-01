from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import uvicorn

# -------------------
# FASTAPI SETUP
# -------------------
app = FastAPI(title="Fake News Detector")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------
# INPUT MODEL
# -------------------
class NewsItem(BaseModel):
    text: str

# -------------------
# LOAD MODEL (CPU)
# -------------------
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # CPU only
)

# -------------------
# LABEL MAPPING
# -------------------
LABEL_MAP = {
    "POSITIVE": "REAL",
    "NEGATIVE": "FAKE"
}

# -------------------
# HELPER: GET SOURCES WITHOUT TOKEN
# -------------------
def get_sources(query, max_results=3):
    """Return top URLs from DuckDuckGo search (no API token)."""
    try:
        query_encoded = urllib.parse.quote(query)
        search_url = f"https://duckduckgo.com/html/?q={query_encoded}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []

        for a in soup.find_all("a", class_="result__a", href=True):
            url = a['href']
            if "uddg=" in url:
                url = urllib.parse.unquote(url.split("uddg=")[-1])
            links.append(url)
            if len(links) >= max_results:
                break

        return links if links else ["No sources found"]
    except Exception as e:
        return [f"Error fetching URLs: {str(e)}"]

# -------------------
# ROUTES
# -------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect(item: NewsItem):
    """Detect if text is FAKE or REAL with dynamic URLs."""
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

        # Dynamic sources based on prediction
        query_type = "fake news" if label == "FAKE" else "news"
        sources = get_sources(f"{query_type} {item.text}")

        return {
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "sources": sources
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------
# RUN APP (PORT FROM RENDER)
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
