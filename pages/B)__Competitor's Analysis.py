

###
#Use T5-small if:
#You want concise and more predictable outputs.
#You prefer a task-specific model where you can conditionally provide inputs (e.g., “The app helps with [phrases].”).
#Use GPT-2 if:
#You prioritize creativity and flexibility in text generation.
#You have sufficient computational resources.

import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from google_play_scraper import reviews,Sort
from wordcloud import WordCloud
import seaborn as sns
from textblob import TextBlob
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from transformers import pipeline
import torch
nltk.download('punkt')

st.set_page_config(page_title="Ratings_and_Reviews", page_icon="✉️")
st.title("Ratings & Reviews of Competitor's App")
st.write("This is Wifi Map's own Ratings and Reviews")
st.header("About This Page")
st.write("This page shows us the analysis of the App's Competitor and shows the exploratory analysis of most recent 200 ratings and reviews.")
# Fetching reviews for the Instabridge app
app_id = "io.wifimap.wifimap"  # App ID for Wifimap on Google Play
result, continuation_token = reviews(
    app_id,
    lang='en',  
    country='us',
    count=200,  
    sort=Sort.NEWEST
    )

reviews_data = [
    {
        "Rating": review['score'],
        "Review": review['content'],
        "ThumbsUpCount": review['thumbsUpCount'],
        "Date": review['at'].strftime('%Y-%m-%d %H:%M:%S'),  
    }
    for review in result
]

reviews_df = pd.DataFrame(reviews_data)
st.dataframe(reviews_df)
reviews_df.to_csv("../wifimap_reviews.csv", index=False)

stop_words = set(stopwords.words('english'))
def calculate_sentiment(review):
    analysis = TextBlob(review)
    sentiment_score = analysis.sentiment.polarity  # Range: -1 to 1
    sentiment_category = 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
    return sentiment_score, sentiment_category

reviews_df['SentimentScore'], reviews_df['SentimentCategory'] = zip(*reviews_df['Review'].map(calculate_sentiment))
# Positive and Negative Reviews
positive_reviews = " ".join(reviews_df[reviews_df['SentimentCategory'] == 'positive']['Review'])
negative_reviews = " ".join(reviews_df[reviews_df['SentimentCategory'] == 'negative']['Review'])

mean_rating = reviews_df['Rating'].mean()
st.subheader(f"Average of Recent 200 Ratings: {mean_rating:.2f}")

# Word Clouds
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=18)
    plt.axis('off')
    st.pyplot(plt) 
    
generate_wordcloud(positive_reviews, "Positive Sentiment Word Cloud from Ratings & Reviews of Wifi Map")
generate_wordcloud(negative_reviews, "Negative Sentiment Word Cloud from Ratings & Reviews of Wifi Map")

# Bar Chart for Sentiment Distribution
sentiment_counts = reviews_df['SentimentCategory'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution', fontsize=18)
plt.xlabel('Sentiment Category', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
st.pyplot(plt) 

# Ratings Distribution
plt.figure(figsize=(8, 6))
sns.histplot(reviews_df['Rating'], bins=5, kde=True)
plt.title('Ratings Distribution', fontsize=18)
plt.xlabel('Ratings', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
st.pyplot(plt)

# Keywords Analysis for Positive and Negative Reviews
def extract_keywords(text, top_n=20):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return Counter(filtered_tokens).most_common(top_n)

positive_keywords = [word for word, _ in extract_keywords(positive_reviews)]
negative_keywords = [word for word, _ in extract_keywords(negative_reviews)]

# Visualize Keywords
def plot_keywords(keywords, title):
    if not keywords:
        print(f"No keywords to plot for: {title}")
        return
    
    try:
        words, counts = zip(*keywords)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(counts), y=list(words), palette="viridis")
        plt.title(title, fontsize=18)
        plt.xlabel('Frequency', fontsize=14)
        plt.ylabel('Keywords', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        st.pyplot(plt)
    except Exception as e:
        print(f"Error plotting {title}: {e}")
        
positive_keywords = extract_keywords(positive_reviews)
negative_keywords = extract_keywords(negative_reviews)

plot_keywords(positive_keywords, "Top Keywords in Positive Reviews")
plot_keywords(negative_keywords, "Top Keywords in Negative Reviews")

st.subheader("Summary Generator")
max_reviews = 100
summarizer = pipeline("summarization")
all_reviews = " ".join(reviews_df['Review'][:max_reviews])
summary = summarizer(all_reviews, max_length=200, min_length=35, do_sample=False)
print("Abstractive Summary:")
st.write(summary[0]['summary_text'])




