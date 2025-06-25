from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from main import my_main
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GSM Operatör Analiz API",
             description="Türk GSM operatörleri için duygu analizi ve varlık çıkarma API'si",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

def clean_format(text: str) -> str:
    """Clean and format text by removing brackets and quotes."""
    return text.replace("[", "").replace("]", "").replace("'", "").strip()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/process_sentence", response_class=HTMLResponse)
async def process_sentence(request: Request, sentence: str = Form(...)):
    """Process the input sentence and return analysis results."""
    try:
        if not sentence or not sentence.strip():
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Lütfen bir yorum giriniz."
            })

        # Process the sentence using my_main function
        data = my_main(sentence)
        
        if not data or "entities" not in data or "sentiment" not in data:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Yorum işlenirken bir hata oluştu. Lütfen tekrar deneyiniz."
            })

        # Clean and format the response
        cleaned_entity_list = [clean_format(entity) for entity in data["entities"]]
        cleaned_sentiment_list = [clean_format(sentiment) for sentiment in data["sentiment"]]

        response_data = {
            "Sentence": sentence,
            "Entities": cleaned_entity_list,
            "Sentiments": cleaned_sentiment_list
        }

        logger.info(f"Successfully processed sentence: {sentence[:50]}...")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "response": response_data
        })

    except Exception as e:
        logger.error(f"Error processing sentence: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Bir hata oluştu. Lütfen daha sonra tekrar deneyiniz."
        })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")