import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from wordcloud import WordCloud
from textblob import TextBlob
import torch
from collections import Counter
import re
from google_play_scraper import app
import spacy
from nltk.tokenize import word_tokenize,sent_tokenize

model_name = "gpt2-medium"  # 'gpt2-medium' or larger for improved quality
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
special_token = ","
tokenizer.add_special_tokens({'pad_token': special_token})
model.resize_token_embeddings(len(tokenizer))

st.subheader("Short Description Generation - Metadata of App")
nlp = spacy.load('en_core_web_sm')
banned_phrases = ["#1", "Best of", "discount", "free for a limited time", "popular", "WiFi"]

def scrape_app_metadata(app_id):
    app_info = app(app_id, lang='en', country='us')
    return {
        "title": app_info['title'],
        "long_description": app_info['description'],
        "features": app_info.get('features', []),
        "category": app_info['genre'],
        "url": app_info['url']
    }

app_id = "com.instabridge.android" #"io.wifimap.wifimap"
metadata = scrape_app_metadata(app_id)
metadata_dataframe = pd.DataFrame([metadata])
    
def extract_key_phrases(text, max_phrases=7):
    doc = nlp(text)
    phrases = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() not in nlp.Defaults.stop_words]
    return phrases[:max_phrases]
    
def generate_text(prompt, max_new_tokens=70, temperature=0.65,num_sequences=3):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # Encoding the prompt
    output = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,  
            num_return_sequences=1, 
            temperature=temperature
        )  # Generating text
        #generated_text = tokenizer.decode(output[0], skip_special_tokens=True) 
        #return generated_text
    num_sequences = min(output.size(0), num_sequences)
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(0,num_sequences)]
    return generated_texts

def generate_summary(text):
        sentences = sent_tokenize(text)
        # Extracting meaningful sentences based on sentiment polarity
        summary = [sent for sent in sentences if TextBlob(sent).sentiment.polarity >= 0.65 and TextBlob(sent).sentiment.polarity <= 0.9][:5]
        st.subheader("Rule-Based Summary:")
        return (" ".join(summary))
    
def remove_banned_phrases(description, banned_phrases):
      for phrase in banned_phrases:
        description = description.replace(phrase, "")
      return description.strip()
  
def truncate_text(text, max_tokens=100):
        input_ids = tokenizer.encode(text, return_tensors="pt")
        truncated_ids = input_ids[:, :max_tokens]  
        return tokenizer.decode(truncated_ids[0], skip_special_tokens=True)
    
def truncate_to_80_characters(description, max_length=80):
        if len(description) <= max_length:
            return description
        truncated = description[:max_length].rsplit(' ', 1)[0]  # Avoiding to cut off mid-word
        return truncated + "..." if len(description) > max_length else truncated


generated_texts = generate_text(truncate_text(metadata["long_description"]), num_sequences=3)
short_descriptions = []

for text in generated_texts:
        summary = generate_summary(text)
        cleaned_summary = remove_banned_phrases(summary, banned_phrases)
        truncated_summary = truncate_to_80_characters(cleaned_summary)
        short_descriptions.append(truncated_summary)
        print(short_descriptions)
        
st.dataframe(metadata_dataframe)
st.header("Phrases from Metadata -> Key Phrases for Short Description Generation")
st.write(extract_key_phrases(metadata["long_description"]))
st.header("Generated Short Descriptions from Metadata of App:")
for i, desc in enumerate(short_descriptions, 0):
        st.write(f"Description Genearted: {desc}")

    #short_description = generate_summary(generate_text(truncate_text(metadata["long_description"])))
    #short_desc = remove_banned_phrases(short_description, banned_phrases)
    #short_desc_final = truncate_to_80_characters(short_desc)
    
    #st.dataframe(metadata_dataframe)
    #st.header("Phrases from Metadata -> Key Phrases for Short Description Generation")
    #st.write(extract_key_phrases(metadata["long_description"]))
    
    #st.header("Generated Short Description from Metadata of App:")
    #st.write(short_desc_final)
    
