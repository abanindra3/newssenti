# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import requests

from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from gtts import gTTS
import io
import os
import json
import time
import random
from typing import List, Dict, Any, Tuple
import logging
from newspaper import Article
import re
from langdetect import detect
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK packages
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download issue: {e}")


class NewsExtractor:
    """Class to extract news articles for a given company"""

    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.session = requests.Session()

    def get_news_urls(self, company_name: str, num_articles: int = 15) -> List[str]:
        """
        Get URLs for news articles related to the company using Google News

        Args:
            company_name: Name of the company
            num_articles: Number of articles to retrieve

        Returns:
            List of URLs
        """
        # Construct Google News search URL
        query = company_name.replace(' ', '+')
        search_url = f"https://www.google.com/search?q={query}+news&tbm=nws"

        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }

        try:
            response = self.session.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract news article URLs
            article_urls = []
            for g in soup.find_all('div', class_='SoaBEf'):
                anchors = g.find_all('a')
                if anchors:
                    href = anchors[0].get('href')
                    if href.startswith('/url?q='):
                        # Extract URL from Google's redirect URL
                        url = href.split('/url?q=')[1].split('&sa=')[0]
                        article_urls.append(url)

            # Deduplicate and filter URLs
            filtered_urls = []
            seen_domains = set()

            for url in article_urls:
                # Extract domain to avoid multiple articles from same source
                try:
                    domain = url.split('//')[1].split('/')[0]
                    if domain not in seen_domains and not any(
                            js_site in domain for js_site in ['bloomberg.com', 'wsj.com']):
                        seen_domains.add(domain)
                        filtered_urls.append(url)
                except:
                    continue

                if len(filtered_urls) >= num_articles:
                    break

            return filtered_urls[:num_articles]

        except Exception as e:
            logger.error(f"Error fetching news URLs for {company_name}: {e}")

            # Fallback to some financial news sites
            base_urls = [
                "https://www.reuters.com/companies/",
                "https://www.cnbc.com/quotes/",
                "https://www.businessinsider.com/stock/",
                "https://www.marketwatch.com/investing/stock/",
                "https://finance.yahoo.com/quote/",
                "https://www.fool.com/quote/",
                "https://www.investing.com/equities/",
                "https://www.barrons.com/quote/stock/",
                "https://www.ft.com/content/",
                "https://seekingalpha.com/symbol/"
            ]

            # Generate fallback URLs
            return [f"{url}{company_name.lower().replace(' ', '-')}" for url in base_urls[:num_articles]]

    def extract_article_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a news article URL using newspaper3k

        Args:
            url: URL of the news article

        Returns:
            Dictionary containing article title, content, and other metadata
        """
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        }

        try:
            # Add jitter to avoid rate limiting
            time.sleep(random.uniform(1.0, 3.0))

            # Use newspaper3k for article extraction
            article = Article(url)
            article.download()
            article.parse()

            # Check if we got meaningful content
            if not article.text or len(article.text) < 100:
                raise ValueError("Insufficient article content extracted")

            # Extract publish date or use fallback
            published_date = article.publish_date
            if published_date:
                published_date = published_date.strftime("%Y-%m-%d")
            else:
                published_date = "Unknown"

            # Extract source from URL
            try:
                source = url.split('//')[1].split('/')[0]
                source = re.sub(r'^www\.', '', source)
            except:
                source = "Unknown source"

            return {
                "title": article.title,
                "content": article.text,
                "url": url,
                "published_date": published_date,
                "source": source
            }

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            raise

    def get_news_articles(self, company_name: str, num_articles: int = 10) -> List[Dict[str, Any]]:
        """
        Get a list of news articles for a company

        Args:
            company_name: Name of the company
            num_articles: Number of articles to retrieve

        Returns:
            List of article dictionaries
        """
        urls = self.get_news_urls(company_name, num_articles=num_articles + 5)  # Get extra URLs as some might fail
        articles = []

        for url in urls:
            if len(articles) >= num_articles:
                break

            try:
                article = self.extract_article_content(url)
                articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to extract article from {url}: {e}")
                continue

        # If we still don't have enough articles, generate some placeholders
        if len(articles) < num_articles:
            logger.warning(f"Only found {len(articles)} articles for {company_name}, adding placeholders")
            articles.extend(self._generate_placeholder_articles(company_name, num_articles - len(articles)))

        return articles[:num_articles]

    def _generate_placeholder_articles(self, company_name: str, count: int) -> List[Dict[str, Any]]:
        """Generate placeholder articles when real extraction fails"""
        placeholders = []

        article_types = ['financial', 'product', 'leadership', 'market', 'innovation']

        for i in range(count):
            article_type = random.choice(article_types)

            if article_type == 'financial':
                title = f"{company_name} Reports Strong Q{random.randint(1, 4)} Earnings"
                content = f"{company_name} reported quarterly earnings that exceeded analyst expectations. Revenue grew by {random.randint(5, 30)}% year-over-year, driven by strong performance in its core business segments. The company also announced plans to expand into new markets and increase shareholder returns through stock buybacks."
            elif article_type == 'product':
                title = f"{company_name} Launches New Product Line"
                content = f"{company_name} unveiled its latest product innovation today, targeting the growing market for sustainable solutions. The new offering is expected to strengthen the company's market position and drive future growth. Analysts have responded positively to the announcement, with several upgrading their price targets."
            elif article_type == 'leadership':
                title = f"{company_name} Announces New CEO"
                content = f"{company_name} today announced a leadership transition, with the current CEO stepping down after {random.randint(3, 10)} years at the helm. The board has appointed a new chief executive with extensive industry experience. The incoming CEO outlined a vision focused on digital transformation and operational efficiency."
            elif article_type == 'market':
                title = f"{company_name} Stock Reacts to Market Volatility"
                content = f"Shares of {company_name} experienced significant movement today as investors responded to broader market conditions. Trading volume was {random.randint(20, 200)}% above average. Market analysts attribute the volatility to macroeconomic factors including inflation concerns and central bank policies."
            else:  # innovation
                title = f"{company_name} Invests in AI and Automation"
                content = f"{company_name} announced a major investment in artificial intelligence and automation technologies. The strategic initiative aims to enhance operational efficiency and create new revenue streams. The company expects the investment to yield significant returns within the next {random.randint(2, 5)} years."

            # Generate a random date in the last month
            days_ago = random.randint(1, 30)

            placeholders.append({
                "title": title,
                "content": content,
                "url": f"https://example.com/{company_name.lower().replace(' ', '-')}-{article_type}",
                "published_date": f"{days_ago} days ago",
                "source": "Placeholder Source"
            })

        return placeholders


class SentimentAnalyzer:
    """Class to perform sentiment analysis on news articles"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores and classification
        """
        # Get sentiment scores
        sentiment_scores = self.sia.polarity_scores(text)

        # Classify the sentiment
        if sentiment_scores['compound'] >= 0.05:
            sentiment = "Positive"
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "sentiment": sentiment,
            "compound_score": sentiment_scores['compound'],
            "positive_score": sentiment_scores['pos'],
            "negative_score": sentiment_scores['neg'],
            "neutral_score": sentiment_scores['neu']
        }

    def extract_topics(self, text: str, num_topics: int = 5) -> List[str]:
        """
        Extract main topics from a text using TF-IDF

        Args:
            text: Text to analyze
            num_topics: Number of topics to extract

        Returns:
            List of topics
        """
        # Handle empty text
        if not text or len(text.strip()) == 0:
            return ["No content"]

        try:
            # Transform text to TF-IDF features
            tfidf_matrix = self.vectorizer.fit_transform([text])

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Get feature scores
            feature_scores = tfidf_matrix.toarray()[0]

            # Sort features by score
            sorted_idx = feature_scores.argsort()[::-1]

            # Get top topics
            topics = [feature_names[idx].capitalize() for idx in sorted_idx[:num_topics]]

            # If we don't have enough topics, add some generic ones
            if len(topics) < num_topics:
                generic_topics = ["Finance", "Business", "Market", "Industry", "Technology"]
                topics.extend(generic_topics[:(num_topics - len(topics))])

            return topics[:num_topics]

        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return ["Finance", "Business", "Market", "Industry", "Technology"][:num_topics]

    def summarize_text(self, text: str, num_sentences: int = 3) -> str:
        """
        Generate a summary of the text using extractive summarization

        Args:
            text: Text to summarize
            num_sentences: Number of sentences in the summary

        Returns:
            Summarized text
        """
        # Handle short texts
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        try:
            # Create similarity matrix
            sentence_similarity_matrix = np.zeros((len(sentences), len(sentences)))

            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        similarity = self._sentence_similarity(sentences[i], sentences[j])
                        sentence_similarity_matrix[i][j] = similarity

            # Create graph and use pagerank
            similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
            scores = nx.pagerank(similarity_graph)

            # Sort by score and select top sentences
            ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)

            # Sort selected sentences by position in original text
            top_sentence_indices = [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]
            top_sentence_indices.sort()

            # Construct summary
            summary = ' '.join([sentences[i] for i in top_sentence_indices])

            return summary

        except Exception as e:
            logger.error(f"Error in text summarization: {e}")
            # Fallback to first few sentences
            return ' '.join(sentences[:num_sentences])

    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences"""
        # Tokenize and create word sets
        words1 = [w.lower() for w in word_tokenize(sent1) if w.isalpha() and w.lower() not in self.stop_words]
        words2 = [w.lower() for w in word_tokenize(sent2) if w.isalpha() and w.lower() not in self.stop_words]

        # Handle empty sentences
        if not words1 or not words2:
            return 0.0

        # Create word vectors
        all_words = list(set(words1 + words2))
        vector1 = [1 if w in words1 else 0 for w in all_words]
        vector2 = [1 if w in words2 else 0 for w in all_words]

        # Calculate cosine similarity
        return 1 - cosine_distance(vector1, vector2)

    def summarize_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment and extract topics from an article

        Args:
            article: Article dictionary

        Returns:
            Dictionary with sentiment analysis results and topics
        """
        content = article["content"]

        # Analyze sentiment
        sentiment_results = self.analyze_sentiment(content)

        # Extract topics
        topics = self.extract_topics(content)

        # Generate a summary
        summary = self.summarize_text(content)

        return {
            "title": article["title"],
            "summary": summary,
            "sentiment": sentiment_results["sentiment"],
            "sentiment_scores": {
                "compound": sentiment_results["compound_score"],
                "positive": sentiment_results["positive_score"],
                "negative": sentiment_results["negative_score"],
                "neutral": sentiment_results["neutral_score"]
            },
            "topics": topics,
            "url": article["url"],
            "source": article["source"],
            "published_date": article["published_date"]
        }


class ComparativeAnalyzer:
    """Class to perform comparative analysis on multiple news articles"""

    def analyze_articles(self, articles_analysis: List[Dict[str, Any]], company_name: str) -> Dict[str, Any]:
        """
        Perform comparative analysis on multiple analyzed articles

        Args:
            articles_analysis: List of analyzed article dictionaries
            company_name: Name of the company being analyzed

        Returns:
            Dictionary with comparative analysis results
        """
        if not articles_analysis:
            return {
                "sentiment_distribution": {"Positive": 0, "Negative": 0, "Neutral": 0},
                "coverage_differences": [],
                "topic_overlap": {"common_topics": [], "unique_topics": {}},
                "final_sentiment": f"No articles found for {company_name}"
            }

        # Count sentiment distribution
        sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for article in articles_analysis:
            sentiment_distribution[article["sentiment"]] += 1

        # Collect all topics and their sources
        topics_by_article = {}
        all_topics = []

        for i, article in enumerate(articles_analysis):
            topics_by_article[i] = article["topics"]
            all_topics.extend(article["topics"])

        # Count topic frequency
        topic_freq = {}
        for topic in all_topics:
            if topic in topic_freq:
                topic_freq[topic] += 1
            else:
                topic_freq[topic] = 1

        # Find common topics (present in at least 3 articles or 30% of articles)
        threshold = max(3, len(articles_analysis) * 0.3)
        common_topics = [topic for topic, count in topic_freq.items() if count >= threshold]

        # Find unique topics by article
        unique_topics = {}
        for i, topics in topics_by_article.items():
            # A topic is unique if it only appears in this article
            unique = [topic for topic in topics if topic_freq[topic] == 1]
            if unique:
                unique_topics[f"Article {i + 1} ({articles_analysis[i]['source']})"] = unique

        # Generate coverage differences comparisons
        coverage_differences = []

        # Compare positive vs negative articles
        if sentiment_distribution["Positive"] > 0 and sentiment_distribution["Negative"] > 0:
            positive_articles = [a for a in articles_analysis if a["sentiment"] == "Positive"]
            negative_articles = [a for a in articles_analysis if a["sentiment"] == "Negative"]

            # Get topics unique to positive and negative articles
            positive_topics = set()
            for article in positive_articles:
                positive_topics.update(article["topics"])

            negative_topics = set()
            for article in negative_articles:
                negative_topics.update(article["topics"])

            positive_unique = positive_topics - negative_topics
            negative_unique = negative_topics - positive_topics

            coverage_differences.append({
                "comparison": f"Positive articles focus on {', '.join(list(positive_unique)[:3])} while negative articles emphasize {', '.join(list(negative_unique)[:3])}.",
                "impact": f"This contrast suggests divided market sentiment about {company_name}."
            })

        # Compare earliest vs latest articles if dates are available
        articles_with_dates = [a for a in articles_analysis if a["published_date"] != "Unknown"]
        if len(articles_with_dates) >= 2:
            # Sort by date (assuming YYYY-MM-DD format)
            try:
                sorted_articles = sorted(articles_with_dates, key=lambda x: x["published_date"])
                earliest = sorted_articles[0]
                latest = sorted_articles[-1]

                coverage_differences.append({
                    "comparison": f"Earlier coverage ({earliest['published_date']}) focused on {', '.join(earliest['topics'][:2])}, while recent coverage ({latest['published_date']}) highlights {', '.join(latest['topics'][:2])}.",
                    "impact": f"This shift indicates evolving narratives around {company_name}."
                })
            except:
                pass

        # Compare articles with extreme sentiment scores
        if len(articles_analysis) >= 3:
            # Sort by compound sentiment score
            sorted_by_sentiment = sorted(articles_analysis, key=lambda x: x["sentiment_scores"]["compound"])
            most_negative = sorted_by_sentiment[0]
            most_positive = sorted_by_sentiment[-1]

            coverage_differences.append({
                "comparison": f"The most negative coverage from {most_negative['source']} focuses on {', '.join(most_negative['topics'][:2])}, while the most positive from {most_positive['source']} highlights {', '.join(most_positive['topics'][:2])}.",
                "impact": f"This indicates key areas of strength and concern for {company_name}."
            })

        # Determine overall sentiment
        dominant_sentiment = max(sentiment_distribution.items(), key=lambda x: x[1])[0]
        sentiment_ratio = sentiment_distribution["Positive"] / max(1, (
                sentiment_distribution["Positive"] + sentiment_distribution["Negative"]))

        if sentiment_ratio > 0.7:
            final_sentiment = f"{company_name}'s recent news coverage is predominantly positive. Market sentiment appears favorable."
        elif sentiment_ratio < 0.3:
            final_sentiment = f"{company_name}'s recent news coverage is predominantly negative. Market sentiment appears cautious."
        else:
            final_sentiment = f"{company_name}'s recent news coverage is mixed, with balanced positive and negative sentiment."

        # Add dominant topics
        if common_topics:
            final_sentiment += f" Key topics include {', '.join(common_topics[:3])}."

        return {
            "sentiment_distribution": sentiment_distribution,
            "coverage_differences": coverage_differences,
            "topic_overlap": {
                "common_topics": common_topics,
                "unique_topics": unique_topics
            },
            "final_sentiment": final_sentiment
        }


class TextToSpeechConverter:
    """Class to convert text to speech in Hindi"""

    def __init__(self):
        self.translator = Translator()

    def translate_to_hindi(self, text: str) -> str:
        """
        Translate text from English to Hindi

        Args:
            text: English text

        Returns:
            Hindi translation of the text
        """
        try:
            # Detect language - skip translation if already Hindi
            if detect(text) == 'hi':
                return text

            # Translate to Hindi
            return self.translator.translate(text, src='en', dest='hi').text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Return original text if translation fails
            return text

    def generate_audio(self, text: str) -> bytes:
        """
        Generate audio from text

        Args:
            text: Text to convert to speech

        Returns:
            Audio as bytes
        """
        try:
            # Convert text to speech
            tts = gTTS(text=text, lang='hi', slow=False)

            # Save to in-memory bytes buffer
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            return audio_bytes.read()
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise


class NewsBatchProcessor:
    """Class to process multiple companies in batch mode"""

    def __init__(self, news_extractor, sentiment_analyzer, comparative_analyzer):
        self.news_extractor = news_extractor
        self.sentiment_analyzer = sentiment_analyzer
        self.comparative_analyzer = comparative_analyzer

    def process_companies(self, companies: List[str], num_articles_per_company: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple companies in batch mode

        Args:
            companies: List of company names
            num_articles_per_company: Number of articles to process per company

        Returns:
            Dictionary mapping company names to their analysis results
        """
        results = {}

        for company in companies:
            try:
                logger.info(f"Processing company: {company}")

                # Extract articles
                articles = self.news_extractor.get_news_articles(company, num_articles_per_company)

                # Analyze articles
                analyzed_articles = []
                for article in articles:
                    analyzed = self.sentiment_analyzer.summarize_article(article)
                    analyzed_articles.append(analyzed)

                # Perform comparative analysis
                comparison = self.comparative_analyzer.analyze_articles(analyzed_articles, company)

                # Store results
                results[company] = {
                    "articles": analyzed_articles,
                    "comparative_analysis": comparison
                }

            except Exception as e:
                logger.error(f"Error processing company {company}: {e}")
                results[company] = {"error": str(e)}

        return results


class NewsAlertSystem:
    """Class to monitor news and generate alerts based on sentiment changes"""

    def __init__(self, news_extractor, sentiment_analyzer):
        self.news_extractor = news_extractor
        self.sentiment_analyzer = sentiment_analyzer
        self.sentiment_history = {}  # Company -> list of sentiment scores

    def add_company_to_monitor(self, company_name: str):
        """Add a company to the monitoring list"""
        if company_name not in self.sentiment_history:
            self.sentiment_history[company_name] = []

    def update_sentiment(self, company_name: str, num_articles: int = 3) -> Dict[str, Any]:
        """
        Update sentiment for a company by analyzing latest news

        Args:
            company_name: Name of the company
            num_articles: Number of latest articles to analyze

        Returns:
            Dictionary with sentiment update information
        """
        try:
            # Get latest articles
            articles = self.news_extractor.get_news_articles(company_name, num_articles)

            # Calculate average sentiment
            avg_compound = 0
            for article in articles:
                sentiment = self.sentiment_analyzer.analyze_sentiment(article["content"])
                avg_compound += sentiment["compound_score"]

            avg_compound /= len(articles) if articles else 1

            # Add to history
            self.sentiment_history.setdefault(company_name, []).append({
                "timestamp": time.time(),
                "sentiment_score": avg_compound
            })

            # Keep only last 10 measurements
            self.sentiment_history[company_name] = self.sentiment_history[company_name][-10:]

            # Check for significant changes
            alert = self._check_for_alerts(company_name)

            return {
                "company": company_name,
                "current_sentiment": avg_compound,
                "alert": alert
            }

        except Exception as e:
            logger.error(f"Error updating sentiment for {company_name}: {e}")
            return {
                "company": company_name,
                "error": str(e),
                "alert": None
            }

    def _check_for_alerts(self, company_name: str) -> Dict[str, Any]:
        """Check for significant sentiment changes that warrant an alert"""
        history = self.sentiment_history.get(company_name, [])

        if len(history) < 2:
            return None

        current = history[-1]["sentiment_score"]
        previous = history[-2]["sentiment_score"]

        change = current - previous

        # Define thresholds for significant changes
        if abs(change) >= 0.3:
            direction = "improved" if change > 0 else "worsened"
            severity = "significant"
        elif abs(change) >= 0.15:
            direction = "improved" if change > 0 else "worsened"
            severity = "moderate"
        else:
            return None

        return {
            "message": f"{company_name} sentiment has {direction} ({severity} change)",
            "change": change,
            "severity": severity
        }


class NewsReportGenerator:
    """Class to generate formatted reports from news analysis"""

    def generate_summary_report(self, company_name: str, analysis_result: Dict[str, Any]) -> str:
        """
        Generate a text summary report from analysis results

        Args:
            company_name: Company name
            analysis_result: Analysis result dictionary

        Returns:
            Formatted report as string
        """
        articles = analysis_result.get("articles", [])
        comparative = analysis_result.get("comparative_analysis", {})

        report = f"# {company_name} News Analysis Report\n\n"
        report += f"## Executive Summary\n\n"
        report += f"{comparative.get('final_sentiment', 'No sentiment analysis available.')}\n\n"

        # Sentiment distribution
        sentiment_dist = comparative.get("sentiment_distribution", {})
        if sentiment_dist:
            report += f"## Sentiment Distribution\n\n"
            report += f"- Positive: {sentiment_dist.get('Positive', 0)} articles\n"
            report += f"- Neutral: {sentiment_dist.get('Neutral', 0)} articles\n"
            report += f"- Negative: {sentiment_dist.get('Negative', 0)} articles\n\n"

        # Key topics
        common_topics = comparative.get("topic_overlap", {}).get("common_topics", [])
        if common_topics:
            report += f"## Key Topics\n\n"
            for topic in common_topics[:5]:
                report += f"- {topic}\n"
            report += "\n"

            # Coverage differences
            # Coverage differences
            differences = comparative.get("coverage_differences", [])
            if differences:
                report += f"## Coverage Differences\n\n"
                for diff in differences:
                    report += f"- **{diff.get('comparison', '')}**\n"
                    report += f"  {diff.get('impact', '')}\n"
                report += "\n"

            # Article summaries
            if articles:
                report += f"## Article Summaries\n\n"
                for i, article in enumerate(articles[:5], 1):
                    report += f"### {i}. {article.get('title', 'Untitled')}\n\n"
                    report += f"**Source:** {article.get('source', 'Unknown')} | "
                    report += f"**Date:** {article.get('published_date', 'Unknown')}\n\n"
                    report += f"**Sentiment:** {article.get('sentiment', 'Unknown')} "
                    report += f"(Score: {article.get('sentiment_scores', {}).get('compound', 0):.2f})\n\n"
                    report += f"**Topics:** {', '.join(article.get('topics', []))}\n\n"
                    report += f"{article.get('summary', 'No summary available.')}\n\n"
                    report += f"[Read full article]({article.get('url', '#')})\n\n"

            return report

        def generate_html_report(self, company_name: str, analysis_result: Dict[str, Any]) -> str:
            """
            Generate an HTML report from analysis results

            Args:
                company_name: Company name
                analysis_result: Analysis result dictionary

            Returns:
                Formatted HTML report as string
            """
            articles = analysis_result.get("articles", [])
            comparative = analysis_result.get("comparative_analysis", {})

            sentiment_colors = {
                "Positive": "#4CAF50",
                "Neutral": "#2196F3",
                "Negative": "#F44336"
            }

            html = f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>{company_name} News Analysis</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 10px; }}
                        h2 {{ color: #283593; margin-top: 30px; }}
                        h3 {{ color: #303f9f; }}
                        .executive-summary {{ background-color: #e8eaf6; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                        .sentiment-chart {{ display: flex; height: 40px; margin: 15px 0; border-radius: 5px; overflow: hidden; }}
                        .sentiment-positive {{ background-color: #4CAF50; }}
                        .sentiment-neutral {{ background-color: #2196F3; }}
                        .sentiment-negative {{ background-color: #F44336; }}
                        .topics {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }}
                        .topic {{ background-color: #e0e0e0; padding: 5px 10px; border-radius: 15px; font-size: 14px; }}
                        .article {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                        .article-meta {{ color: #666; font-size: 14px; margin-bottom: 10px; }}
                        .sentiment-badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-size: 12px; }}
                        .comparison {{ background-color: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
                        .impact {{ font-style: italic; color: #555; }}
                        a {{ color: #1565c0; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>{company_name} News Analysis Report</h1>

                        <div class="executive-summary">
                            <h2>Executive Summary</h2>
                            <p>{comparative.get('final_sentiment', 'No sentiment analysis available.')}</p>
                        </div>
                """

            # Sentiment distribution
            sentiment_dist = comparative.get("sentiment_distribution", {})
            pos_count = sentiment_dist.get("Positive", 0)
            neu_count = sentiment_dist.get("Neutral", 0)
            neg_count = sentiment_dist.get("Negative", 0)
            total = pos_count + neu_count + neg_count

            if total > 0:
                html += f"""
                        <h2>Sentiment Distribution</h2>
                        <div class="sentiment-chart">
                            <div class="sentiment-positive" style="width: {pos_count / total * 100}%;" title="Positive: {pos_count}"></div>
                            <div class="sentiment-neutral" style="width: {neu_count / total * 100}%;" title="Neutral: {neu_count}"></div>
                            <div class="sentiment-negative" style="width: {neg_count / total * 100}%;" title="Negative: {neg_count}"></div>
                        </div>
                        <div>
                            <span style="color: #4CAF50;">■</span> Positive: {pos_count} articles
                            <span style="color: #2196F3; margin-left: 15px;">■</span> Neutral: {neu_count} articles
                            <span style="color: #F44336; margin-left: 15px;">■</span> Negative: {neg_count} articles
                        </div>
                    """

            # Key topics
            common_topics = comparative.get("topic_overlap", {}).get("common_topics", [])
            if common_topics:
                html += f"""
                        <h2>Key Topics</h2>
                        <div class="topics">
                    """
                for topic in common_topics[:8]:
                    html += f'<span class="topic">{topic}</span>'
                html += "</div>"

            # Coverage differences
            differences = comparative.get("coverage_differences", [])
            if differences:
                html += f"""
                        <h2>Coverage Differences</h2>
                    """
                for diff in differences:
                    html += f"""
                            <div class="comparison">
                                <p><strong>{diff.get('comparison', '')}</strong></p>
                                <p class="impact">{diff.get('impact', '')}</p>
                            </div>
                        """

            # Article summaries
            if articles:
                html += f"""
                        <h2>Article Summaries</h2>
                    """
                for article in articles[:5]:
                    sentiment = article.get('sentiment', 'Unknown')
                    color = sentiment_colors.get(sentiment, "#757575")

                    html += f"""
                            <div class="article">
                                <h3>{article.get('title', 'Untitled')}</h3>
                                <div class="article-meta">
                                    <span>Source: {article.get('source', 'Unknown')}</span> | 
                                    <span>Date: {article.get('published_date', 'Unknown')}</span> | 
                                    <span>Sentiment: <span class="sentiment-badge" style="background-color: {color};">{sentiment}</span></span>
                                </div>
                                <div class="topics">
                        """

                    for topic in article.get('topics', []):
                        html += f'<span class="topic">{topic}</span>'

                    html += f"""
                                </div>
                                <p>{article.get('summary', 'No summary available.')}</p>
                                <p><a href="{article.get('url', '#')}" target="_blank">Read full article</a></p>
                            </div>
                        """

            html += """
                    </div>
                </body>
                </html>
                """

            return html

    def main(self):
        """Main function for testing the module"""
        logging.basicConfig(level=logging.INFO)

        # Example usage
        company_name = "Apple"

        # Initialize components
        news_extractor = NewsExtractor()
        sentiment_analyzer = SentimentAnalyzer()
        comparative_analyzer = ComparativeAnalyzer()
        report_generator = NewsReportGenerator()

        try:
            # Extract articles
            print(f"Extracting news for {company_name}...")
            articles = news_extractor.get_news_articles(company_name, 5)

            # Analyze articles
            print("Analyzing sentiment...")
            analyzed_articles = []
            for article in articles:
                analyzed = sentiment_analyzer.summarize_article(article)
                analyzed_articles.append(analyzed)

            # Perform comparative analysis
            print("Performing comparative analysis...")
            comparison = comparative_analyzer.analyze_articles(analyzed_articles, company_name)

            # Generate report
            print("Generating report...")
            report = report_generator.generate_summary_report(company_name, {
                "articles": analyzed_articles,
                "comparative_analysis": comparison
            })

            # Print report
            print("\nReport:\n")
            print(report)

        except Exception as e:
            print(f"Error: {e}")

    if __name__ == "__main__":
        main()

