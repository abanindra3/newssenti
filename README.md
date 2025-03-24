News Summarization and Text-to-Speech Application
Overview
This application extracts key details from multiple news articles related to a given company, performs sentiment analysis, conducts a comparative analysis, and generates a text-to-speech (TTS) output in Hindi. The tool allows users to input a company name and receive a structured sentiment report along with an audio output.
Key Features

📰 News Extraction: Extracts title, summary, and metadata from news articles related to a company
📊 Sentiment Analysis: Performs sentiment analysis (positive, negative, neutral) on article content
🔍 Comparative Analysis: Conducts comparative sentiment analysis across articles
🔊 Text-to-Speech: Converts summarized content into Hindi speech
🌐 Web Interface: Simple web-based UI built with Streamlit
🔄 API Backend: Communication between frontend and backend via RESTful APIs

Architecture
The application follows a client-server architecture:

Frontend: Streamlit-based web interface (app.py)
Backend API: FastAPI-based RESTful API (api.py)
Core Utilities: News extraction, sentiment analysis, and TTS conversion (utils.py)

Installation and Setup
Prerequisites

Python 3.8 or higher
Internet connection for news extraction and API access

Installation

Clone this repository:
bashCopygit clone https://github.com/yourusername/news-summarization-app.git
cd news-summarization-app

Create a virtual environment:
bashCopypython -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install required packages:
bashCopypip install -r requirements.txt

Download NLTK data:
pythonCopypython -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"


Running the Application

Start the API server:
bashCopyuvicorn api:app --host 0.0.0.0 --port 8000 --reload

In a separate terminal, start the Streamlit frontend:
bashCopystreamlit run app.py

Open your browser and navigate to:

Frontend UI: http://localhost:8501
API Documentation: http://localhost:8000/docs



API Usage
Core Endpoints

POST /api/analyze - Analyze news for a company
jsonCopy{
  "company_name": "Tesla",
  "num_articles": 10,
  "force_refresh": false
}

GET /api/audio/{audio_id} - Get the generated audio file
GET /api/companies - Get a list of common companies to analyze

Using the API with Postman

Launch Postman
Create a new POST request to http://localhost:8000/api/analyze
Set Body to raw JSON and provide:
jsonCopy{
  "company_name": "Tesla",
  "num_articles": 10
}

Send the request and examine the response
To get the audio, use the URL from audio_url in a GET request

Models and Implementation Details
News Extraction

Uses a combination of web scraping (BeautifulSoup) and the newspaper3k library
Extracts news from multiple sources to provide diverse perspectives
Implements fallbacks and error handling for robust operation

Sentiment Analysis

Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment scoring
Extracts topics using TF-IDF vectorization
Generates summaries using extractive summarization based on PageRank algorithm

Comparative Analysis

Compares sentiment distribution across articles
Identifies common and unique topics
Analyzes coverage differences and their potential impact

Text-to-Speech

Translates English text to Hindi using Google Translate
Converts Hindi text to speech using gTTS (Google Text-to-Speech)

Hugging Face Space Deployment
The application is deployed on Hugging Face Spaces and can be accessed at:
https://huggingface.co/spaces/yourusername/news-sentiment-app
Assumptions and Limitations

News Extraction: Limited to non-JavaScript websites that can be scraped using BeautifulSoup
Language: Primary analysis is done in English with only the final summary translated to Hindi
API Rate Limits: May encounter rate limiting when making multiple requests to news sources
Text-to-Speech: Quality depends on the Google TTS service

Future Improvements

Implement more advanced NLP techniques for better sentiment analysis
Add support for more languages
Improve topic extraction using named entity recognition
Implement better caching strategies for improved performance
Add user authentication for personalized experiences
