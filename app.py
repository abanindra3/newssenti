#hi
'''
import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import time
import base64
from io import BytesIO
from typing import Dict, List, Any

# Set page configuration
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint (change this when deployed)
API_ENDPOINT = "https://newssenti.onrender.com/"

# App title and description
st.title("📰 News Sentiment Analysis & TTS")
st.markdown("""
This application extracts news articles for a company, performs sentiment analysis, 
and provides a comparative analysis along with a Hindi text-to-speech summary.
""")


# Function to fetch common companies
@st.cache_data(ttl=3600)
def get_common_companies():
    try:
        response = requests.get(f"{API_ENDPOINT}/api/companies")
        return response.json()["companies"]
    except Exception as e:
        st.error(f"Error fetching companies: {e}")
        return ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]


# Function to analyze company news
def analyze_company(company_name: str, num_articles: int, force_refresh: bool):
    try:
        payload = {
            "company_name": company_name,
            "num_articles": num_articles,
            "force_refresh": force_refresh
        }

        with st.spinner(f"Analyzing {company_name} news. This may take a minute..."):
            response = requests.post(f"{API_ENDPOINT}/api/analyze", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error analyzing company: {e}")
        return None


# Function to create a sentiment distribution chart
def create_sentiment_chart(sentiment_distribution: Dict[str, int]):
    # Create dataframe
    df = pd.DataFrame({
        "Sentiment": list(sentiment_distribution.keys()),
        "Count": list(sentiment_distribution.values())
    })

    # Define colors
    colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}

    # Create chart
    fig = px.pie(
        df,
        values="Count",
        names="Sentiment",
        title="News Sentiment Distribution",
        color="Sentiment",
        color_discrete_map=colors,
        hole=0.4
    )

    # Update layout
    fig.update_layout(
        legend_title="Sentiment",
        font=dict(size=14),
        margin=dict(t=50, b=20, l=20, r=20)
    )

    return fig


# Function to display a heatmap of article topics
def create_topic_heatmap(articles: List[Dict[str, Any]]):
    # Extract topics from articles
    all_topics = []
    for article in articles:
        all_topics.extend(article["topics"])

    # Count unique topics
    unique_topics = list(set(all_topics))

    # Create a matrix of topic presence in each article
    matrix = []
    for i, article in enumerate(articles):
        row = []
        for topic in unique_topics:
            row.append(1 if topic in article["topics"] else 0)
        matrix.append(row)

    # Create dataframe
    df = pd.DataFrame(matrix, columns=unique_topics)

    # Get sources for y-axis labels
    sources = [f"{i + 1}. {a['source']}" for i, a in enumerate(articles)]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=unique_topics,
        y=sources,
        colorscale="Blues",
        showscale=False
    ))

    # Update layout
    fig.update_layout(
        title="Topic Distribution Across News Sources",
        xaxis_title="Topics",
        yaxis_title="News Sources",
        margin=dict(t=50, b=100, l=100, r=20),
        xaxis=dict(tickangle=-45),
        height=500
    )

    return fig


# Sidebar for company selection
st.sidebar.header("Company Selection")

# Get common companies and add custom option
common_companies = get_common_companies()
company_options = ["Custom"] + common_companies

selected_option = st.sidebar.selectbox(
    "Choose a company",
    options=company_options
)

# Get company name (either from selection or custom input)
if selected_option == "Custom":
    company_name = st.sidebar.text_input("Enter company name", "")
else:
    company_name = selected_option

# Additional options
num_articles = st.sidebar.slider(
    "Number of articles",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

force_refresh = st.sidebar.checkbox("Force refresh (ignore cache)", value=False)

# Main content
if company_name:
    if st.sidebar.button("Analyze News"):
        # Call API to analyze company
        result = analyze_company(company_name, num_articles, force_refresh)

        if result:
            # Store result in session state for reuse
            st.session_state.analysis_result = result

    # Display analysis results if available
    if "analysis_result" in st.session_state:
        result = st.session_state.analysis_result

        # Company overview
        st.header(f"📊 {result['company']} News Analysis")

        # Create columns for charts
        col1, col2 = st.columns([1, 2])

        with col1:
            # Sentiment distribution
            st.plotly_chart(
                create_sentiment_chart(result["sentiment_distribution"]),
                use_container_width=True
            )

            # Final sentiment analysis
            st.subheader("Overall Sentiment Analysis")
            st.write(result["final_sentiment"])

            # Audio playback
            st.subheader("Hindi Summary")

            try:
                audio_response = requests.get(f"{API_ENDPOINT}{result['audio_url']}")
                if audio_response.status_code == 200:
                    st.audio(audio_response.content, format="audio/mp3")
                else:
                    st.error("Error loading audio")
            except Exception as e:
                st.error(f"Error playing audio: {e}")

        with col2:
            # Topic heatmap
            st.plotly_chart(
                create_topic_heatmap(result["articles"]),
                use_container_width=True
            )

        # Display coverage differences
        st.subheader("Coverage Differences")
        for i, diff in enumerate(result["coverage_differences"]):
            with st.expander(f"Comparison {i + 1}"):
                st.write(f"**Comparison:** {diff['comparison']}")
                st.write(f"**Impact:** {diff['impact']}")

        # Display news articles
        st.subheader("News Articles")

        # Create tabs for each article
        tabs = st.tabs([f"{i + 1}. {a['source']}" for i, a in enumerate(result["articles"])])

        for i, (tab, article) in enumerate(zip(tabs, result["articles"])):
            with tab:
                # Get sentiment color
                sentiment_color = {
                    "Positive": "green",
                    "Negative": "red",
                    "Neutral": "blue"
                }.get(article["sentiment"], "black")

                # Display article details
                st.markdown(f"## {article['title']}")
                st.markdown(f"**Source:** {article['source']} | **Date:** {article['published_date']} | "
                            f"**Sentiment:** :{sentiment_color}[{article['sentiment']}]")

                st.markdown("### Summary")
                st.write(article["summary"])

                st.markdown("### Topics")
                st.write(", ".join(article["topics"]))

                st.markdown(f"[Read Full Article]({article['url']})")
else:
    st.info("Please select or enter a company name and click 'Analyze News'")

# Footer
st.markdown("---")
st.markdown("**News Sentiment Analysis & TTS Application** | Developed for the assignment")
st.markdown("Developed by [Abanindra](https://github.com/abanindra3)", unsafe_allow_html=True)
'''


import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import time
import base64
from io import BytesIO
from typing import Dict, List, Any

# Set page configuration
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint (change this when deployed)
API_ENDPOINT = "https://newssenti.onrender.com/"

# App title and description
st.title("📰 News Sentiment Analysis & TTS")
st.markdown("""
This application extracts news articles for a company, performs sentiment analysis, 
and provides a comparative analysis along with a Hindi text-to-speech summary.
""")


# Function to fetch common companies
@st.cache(ttl=3600)
def get_common_companies():
    try:
        response = requests.get(f"{API_ENDPOINT}/api/companies")
        return response.json()["companies"]
    except Exception as e:
        st.error(f"Error fetching companies: {e}")
        return ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]


# Function to analyze company news
def analyze_company(company_name: str, num_articles: int, force_refresh: bool):
    try:
        payload = {
            "company_name": company_name,
            "num_articles": num_articles,
            "force_refresh": force_refresh
        }

        with st.spinner(f"Analyzing {company_name} news. This may take a minute..."):
            response = requests.post(f"{API_ENDPOINT}/api/analyze", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error analyzing company: {e}")
        return None


# Function to create a sentiment distribution chart
def create_sentiment_chart(sentiment_distribution: Dict[str, int]):
    # Create dataframe
    df = pd.DataFrame({
        "Sentiment": list(sentiment_distribution.keys()),
        "Count": list(sentiment_distribution.values())
    })

    # Define colors
    colors = {"Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3"}

    # Create chart
    fig = px.pie(
        df,
        values="Count",
        names="Sentiment",
        title="News Sentiment Distribution",
        color="Sentiment",
        color_discrete_map=colors,
        hole=0.4
    )

    # Update layout
    fig.update_layout(
        legend_title="Sentiment",
        font=dict(size=14),
        margin=dict(t=50, b=20, l=20, r=20)
    )

    return fig


# Function to display a heatmap of article topics
def create_topic_heatmap(articles: List[Dict[str, Any]]):
    # Extract topics from articles
    all_topics = []
    for article in articles:
        all_topics.extend(article["topics"])

    # Count unique topics
    unique_topics = list(set(all_topics))

    # Create a matrix of topic presence in each article
    matrix = []
    for i, article in enumerate(articles):
        row = []
        for topic in unique_topics:
            row.append(1 if topic in article["topics"] else 0)
        matrix.append(row)

    # Create dataframe
    df = pd.DataFrame(matrix, columns=unique_topics)

    # Get sources for y-axis labels
    sources = [f"{i + 1}. {a['source']}" for i, a in enumerate(articles)]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=unique_topics,
        y=sources,
        colorscale="Blues",
        showscale=False
    ))

    # Update layout
    fig.update_layout(
        title="Topic Distribution Across News Sources",
        xaxis_title="Topics",
        yaxis_title="News Sources",
        margin=dict(t=50, b=100, l=100, r=20),
        xaxis=dict(tickangle=-45),
        height=500
    )

    return fig


# Sidebar for company selection
st.sidebar.header("Company Selection")

# Get common companies and add custom option
common_companies = get_common_companies()
company_options = ["Custom"] + common_companies

selected_option = st.sidebar.selectbox(
    "Choose a company",
    options=company_options
)

# Get company name (either from selection or custom input)
if selected_option == "Custom":
    company_name = st.sidebar.text_input("Enter company name", "")
else:
    company_name = selected_option

# Additional options
num_articles = st.sidebar.slider(
    "Number of articles",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

force_refresh = st.sidebar.checkbox("Force refresh (ignore cache)", value=False)

# Main content
if company_name:
    if st.sidebar.button("Analyze News"):
        # Call API to analyze company
        result = analyze_company(company_name, num_articles, force_refresh)

        if result:
            # Store result in session state for reuse
            st.session_state.analysis_result = result

    # Display analysis results if available
    if "analysis_result" in st.session_state:
        result = st.session_state.analysis_result

        # Company overview
        st.header(f"📊 {result['company']} News Analysis")

        # Create columns for charts
        col1, col2 = st.columns([1, 2])

        with col1:
            # Sentiment distribution
            st.plotly_chart(
                create_sentiment_chart(result["sentiment_distribution"]),
                use_container_width=True
            )

            # Final sentiment analysis
            st.subheader("Overall Sentiment Analysis")
            st.write(result["final_sentiment"])

            # Audio playback
            st.subheader("Hindi Summary")

            try:
                audio_response = requests.get(f"{API_ENDPOINT}{result['audio_url']}")
                if audio_response.status_code == 200:
                    st.audio(audio_response.content, format="audio/mp3")
                else:
                    st.error("Error loading audio")
            except Exception as e:
                st.error(f"Error playing audio: {e}")

        with col2:
            # Topic heatmap
            st.plotly_chart(
                create_topic_heatmap(result["articles"]),
                use_container_width=True
            )

        # Display coverage differences
        st.subheader("Coverage Differences")
        for i, diff in enumerate(result["coverage_differences"]):
            with st.expander(f"Comparison {i + 1}"):
                st.write(f"**Comparison:** {diff['comparison']}")
                st.write(f"**Impact:** {diff['impact']}")

        # Display news articles
        st.subheader("News Articles")

        # Create tabs for each article
        tabs = st.tabs([f"{i + 1}. {a['source']}" for i, a in enumerate(result["articles"])])

        for i, (tab, article) in enumerate(zip(tabs, result["articles"])):
            with tab:
                # Get sentiment color
                sentiment_color = {
                    "Positive": "green",
                    "Negative": "red",
                    "Neutral": "blue"
                }.get(article["sentiment"], "black")

                # Display article details
                st.markdown(f"## {article['title']}")
                st.markdown(f"**Source:** {article['source']} | **Date:** {article['published_date']} | "
                            f"**Sentiment:** :{sentiment_color}[{article['sentiment']}]")

                st.markdown("### Summary")
                st.write(article["summary"])

                st.markdown("### Topics")
                st.write(", ".join(article["topics"]))

                st.markdown(f"[Read Full Article]({article['url']})")
else:
    st.info("Please select or enter a company name and click 'Analyze News'")

# Footer
st.markdown("---")
st.markdown("**News Sentiment Analysis & TTS Application** | Developed for the assignment")
st.markdown("Developed by [Abanindra](https://github.com/abanindra3)", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**News Sentiment Analysis & TTS Application** | Developed for the assignment")
st.markdown("Developed by [Abanindra](https://github.com/abanindra3)", unsafe_allow_html=True)
