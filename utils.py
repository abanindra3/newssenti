from nltk.sentiment import SentimentIntensityAnalyzer
import requests
import json
from gtts import gTTS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from googletrans import Translator
import io

# Download necessary NLTK packages
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class NewsExtractor:


    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def get_news_articles(self, query, num_articles=5):

        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": num_articles
        }
        response = requests.get(self.base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": article["title"],
                    "content": article["description"] or article["content"],
                    "url": article["url"],
                    "published_date": article["publishedAt"],
                    "source": article["source"]["name"]
                }
                for article in articles if article["content"]
            ]
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return []


class SentimentAnalyzer:
    """Perform sentiment analysis on news articles"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """Analyze the sentiment of a given text"""
        sentiment_scores = self.sia.polarity_scores(text)
        sentiment = "Positive" if sentiment_scores['compound'] > 0.05 else "Negative" if sentiment_scores[
                                                                                             'compound'] < -0.05 else "Neutral"
        return {"sentiment": sentiment, "scores": sentiment_scores}


class ComparativeAnalyzer:
    """Compare multiple news articles and generate overall sentiment"""

    def analyze_articles(self, analyzed_articles):
        """Analyze sentiment trends across multiple articles"""
        if not analyzed_articles:
            return {"sentiment_distribution": {}, "coverage_differences": [], "topic_overlap": {},
                    "final_sentiment": "No articles to compare."}

        # Count sentiment distribution
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for article in analyzed_articles:
            sentiment_counts[article["sentiment"]] += 1

        # Calculate sentiment distribution percentages
        total = len(analyzed_articles)

        # Compare coverage differences
        coverage_differences = []
        sources = [a["source"] for a in analyzed_articles]

        for i in range(min(3, len(analyzed_articles))):
            for j in range(i + 1, min(4, len(analyzed_articles))):
                if i != j:
                    comparison = f"Comparing {sources[i]} vs {sources[j]}"
                    if analyzed_articles[i]["sentiment"] != analyzed_articles[j]["sentiment"]:
                        impact = f"{sources[i]} shows {analyzed_articles[i]['sentiment']} sentiment while {sources[j]} shows {analyzed_articles[j]['sentiment']} sentiment."
                    else:
                        impact = f"Both sources show {analyzed_articles[i]['sentiment']} sentiment, but may differ in intensity."

                    coverage_differences.append({"comparison": comparison, "impact": impact})

        # Extract topics (placeholder - in a real app, you would extract actual topics)
        topic_overlap = {
            "common_topics": ["business", "finance"],
            "unique_topics": {
                "Source 1": ["technology", "innovation"],
                "Source 2": ["markets", "economy"]
            }
        }

        # Generate final sentiment analysis
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        final_sentiment = f"Based on the analysis of {total} articles, the overall sentiment towards the company is {dominant_sentiment.lower()}. "
        final_sentiment += f"{sentiment_counts['Positive']} articles showed positive sentiment, {sentiment_counts['Negative']} negative, and {sentiment_counts['Neutral']} neutral."

        return {
            "sentiment_distribution": sentiment_counts,
            "coverage_differences": coverage_differences,
            "topic_overlap": topic_overlap,
            "final_sentiment": final_sentiment
        }


class TextToSpeechConverter:
    """Convert news text to speech"""

    def __init__(self):
        self.translator = Translator()

    def translate_to_hindi(self, text):
        """Translate English text to Hindi"""
        try:
            translation = self.translator.translate(text, src='en', dest='hi')
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            # Return original text if translation fails
            return f"Hindi translation unavailable. Original text: {text}"

    def generate_audio(self, text, language="hi"):
        """Convert text to speech and return MP3 bytes"""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp.read()
        except Exception as e:
            print(f"TTS error: {e}")
            # Return empty bytes if TTS fails
            return b""


# Define summarizer class to match the import in FastAPI app
class Summarizer:
    def summarize_text(self, text, num_sentences=3):
        """Generate a summary using extractive summarization"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        try:
            # Simple fallback for very short texts
            if len(sentences) < 2:
                return text

            similarity_matrix = np.zeros((len(sentences), len(sentences)))
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform(sentences).toarray()

            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    similarity_matrix[i][j] = np.dot(vectors[i], vectors[j])

            graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(graph)
            ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
            summary = ' '.join(sentences[ranked_sentences[i][1]] for i in range(min(num_sentences, len(sentences))))
            return summary
        except Exception as e:
            print(f"Summarization error: {e}")
            # Return first few sentences if summarization fails
            return ' '.join(sentences[:min(num_sentences, len(sentences))])


# Create an instance of Summarizer to match the import in FastAPI app
summarizer = Summarizer()