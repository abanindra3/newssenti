from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import io
import json
import time
import logging
from dotenv import load_dotenv
import os
from utils import NewsExtractor, SentimentAnalyzer, ComparativeAnalyzer, TextToSpeechConverter, summarizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI(
    title="News Sentiment & TTS API",
    description="API for extracting, analyzing and converting news to speech",
    version="1.0.0"
)

API_KEY = "87c655e52ea4430da0894c0a8cdef473"
#load_dotenv()
#API_KEY = os.getenv("API_KEY")
news_extractor = NewsExtractor(API_KEY)
sentiment_analyzer = SentimentAnalyzer()
comparative_analyzer = ComparativeAnalyzer()
tts_converter = TextToSpeechConverter()

analysis_cache = {}

class CompanyRequest(BaseModel):
    """Requesting the  model for company analysis"""
    company_name: str = Field(..., min_length=1, max_length=100, description="Name of the company to analyze")
    num_articles: Optional[int] = Field(10, ge=1, le=20, description="Number of articles to analyze")
    force_refresh: Optional[bool] = Field(False, description="Force refresh analysis instead of using cache")


class ArticleResponse(BaseModel):
    """Response model for article analysis"""
    title: str
    summary: str
    sentiment: str
    topics: List[str]
    url: str
    source: str
    published_date: str


class ComparisonResponse(BaseModel):
    """Response model for comparison analysis"""
    comparison: str
    impact: str


class SentimentDistributionResponse(BaseModel):
    """Response model for sentiment distribution"""
    positive: int
    negative: int
    neutral: int


class TopicOverlapResponse(BaseModel):
    """Response model for topic overlap analysis"""
    common_topics: List[str]
    unique_topics: Dict[str, List[str]]


class AnalysisResponse(BaseModel):
    """Response model for full analysis"""
    company: str
    articles: List[ArticleResponse]
    sentiment_distribution: Dict[str, int]
    coverage_differences: List[Dict[str, str]]
    topic_overlap: Dict[str, Any]
    final_sentiment: str
    audio_url: str


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_company(request: CompanyRequest, background_tasks: BackgroundTasks):
    """
    Analysing news for a company, and then performing sentiment analysis, and generate TTS
    """
    company_name = request.company_name
    num_articles = request.num_articles

    # Check cache unless force refresh is requested
    cache_key = f"{company_name}_{num_articles}"
    if not request.force_refresh and cache_key in analysis_cache:
        cache_entry = analysis_cache[cache_key]
        # Check if cache is fresh (less than 1 hour old)
        if time.time() - cache_entry["timestamp"] < 3600:
            logger.info(f"Using cached analysis for {company_name}")
            return cache_entry["data"]

    try:
        # Extract news articles
        logger.info(f"Extracting news for {company_name}")
        articles = news_extractor.get_news_articles(company_name, num_articles)

        if not articles:
            raise HTTPException(status_code=404, detail=f"No news articles found for {company_name}")

        # Analyze articles
        logger.info(f"Analyzing sentiment for {len(articles)} articles")
        analyzed_articles = []

        for article in articles:
            summary = summarizer.summarize_text(article["content"])
            sentiment = sentiment_analyzer.analyze_sentiment(article["content"])

            analyzed = {
                "title": article["title"],
                "summary": summary,
                "sentiment": sentiment["sentiment"],
                "sentiment_scores": sentiment["scores"],
                "topics": ["business", "news", "finance"],  # Mock topics for demonstration
                "url": article["url"],
                "source": article["source"],
                "published_date": article["published_date"]
            }

            analyzed_articles.append(analyzed)

        # Performing the  comparative analysis
        logger.info("Performing comparative analysis")
        comparison = comparative_analyzer.analyze_articles(analyzed_articles)

        # Generating Hindi TTS for the final sentiment analysis
        logger.info("Generating Hindi TTS")
        hindi_text = tts_converter.translate_to_hindi(comparison["final_sentiment"])
        audio_bytes = tts_converter.generate_audio(hindi_text)


        audio_id = f"{company_name.lower().replace(' ', '_')}_{int(time.time())}"

        # Store audio bytes in memory (in production, use proper storage)
        app.state.audio_cache = getattr(app.state, "audio_cache", {})
        app.state.audio_cache[audio_id] = audio_bytes

        # Prepare response
        response_data = {
            "company": company_name,
            "articles": [
                {
                    "title": a["title"],
                    "summary": a["summary"],
                    "sentiment": a["sentiment"],
                    "topics": a["topics"],
                    "url": a["url"],
                    "source": a["source"],
                    "published_date": a["published_date"]
                } for a in analyzed_articles
            ],
            "sentiment_distribution": comparison["sentiment_distribution"],
            "coverage_differences": comparison["coverage_differences"],
            "topic_overlap": comparison["topic_overlap"],
            "final_sentiment": comparison["final_sentiment"],
            "audio_url": f"/api/audio/{audio_id}"
        }

        # Store in cache
        analysis_cache[cache_key] = {
            "data": response_data,
            "timestamp": time.time()
        }

        # Schedule cache cleanup in background
        background_tasks.add_task(cleanup_old_cache)

        return response_data

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing news: {str(e)}")


@app.get("/api/audio/{audio_id}")
async def get_audio(audio_id: str):

    audio_cache = getattr(app.state, "audio_cache", {})

    if audio_id not in audio_cache:
        raise HTTPException(status_code=404, detail="Audio not found")

    audio_bytes = audio_cache[audio_id]

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mp3",
        headers={"Content-Disposition": f"attachment; filename={audio_id}.mp3"}
    )


@app.get("/api/companies")
async def get_common_companies():

    common_companies = [
        "Apple", "Microsoft", "Google", "Amazon", "Meta",
        "Tesla", "NVIDIA", "Netflix", "IBM", "Intel",
        "Samsung", "Toyota", "JPMorgan Chase", "Walmart", "Coca-Cola"
    ]

    return {"companies": common_companies}


async def cleanup_old_cache():

    current_time = time.time()
    keys_to_remove = []

    for key, entry in analysis_cache.items():
        # Remove entries older than 24 hours
        if current_time - entry["timestamp"] > 86400:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del analysis_cache[key]


    audio_cache = getattr(app.state, "audio_cache", {})
    keys_to_remove = []

    for key in audio_cache:
        # Audio files are deleted after 1 hour
        if "_" in key:
            timestamp = int(key.split("_")[-1])
            if current_time - timestamp > 3600:
                keys_to_remove.append(key)

    for key in keys_to_remove:
        del audio_cache[key]


@app.get("/")
async def root():

    return {
        "name": "News Sentiment & TTS API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/api/analyze", "method": "POST", "description": "Analyze news for a company"},
            {"path": "/api/audio/{audio_id}", "method": "GET", "description": "Get generated audio file"},
            {"path": "/api/companies", "method": "GET", "description": "Get list of common companies"}
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
