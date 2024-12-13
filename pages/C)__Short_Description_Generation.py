import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline
from wordcloud import WordCloud
import torch
from collections import Counter
import re
from google_play_scraper import app
import spacy
#import thinc

# Loading pre-trained GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2-medium"  # 'gpt2-medium' or larger for improved quality
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
special_token = ","
tokenizer.add_special_tokens({'pad_token': special_token})
model.resize_token_embeddings(len(tokenizer))

st.set_page_config(
    page_title = 'AB Test Short Description Generartion',
    page_icon = '✅',
    layout = 'wide')

st.header("About This Page")
st.write("This page shows us two ways of generating statements, majorly Short Description. One from Historic Data of different store listings participating in previous A/B testing of app/game, another is generating short description from the Meta Data of app/game")

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

if st.checkbox("Short Description Generation - Past A/B Test Results"):
    
    st.subheader("File Uploader and Data Viewer - A/B Testing Hypothesis Recommendation")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):  # Checking if the file is CSV or Excel
            df = pd.read_csv(uploaded_file)      # Reading CSV file
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)    # Reading excel file
        st.write(f"File name: **{uploaded_file.name}**")
      
        df["Start Date"] = pd.to_datetime(df["Start Date"])
        df["End Date"] = pd.to_datetime(df["End Date"])
        
        df["1st Time Installer's Performance"] = (df['Performance (90% confidence interval)'] + df['Performance (90% confidence interval).1'])/2
        df["Retained Installer's Performance"] = (df['Performance (90% confidence interval).2'] + df['Performance (90% confidence interval).3'])/2
        df["Experiment Date"] = df["Start Date"].dt.strftime('%Y-%m-%d') + " - " + df["End Date"].dt.strftime('%Y-%m-%d')
        df["1st Time Installer's Performance"] = df["1st Time Installer's Performance"].astype(str)
        df["Performance (%)"] = df["1st Time Installer's Performance"].str.rstrip('%').astype(float)
        df["Performance of Conversion Rate (%)"] = df.groupby("Experiment Date")["Performance (%)"].diff()
        df["Performance of Conversion Rate (%)"] = df["Performance of Conversion Rate (%)"].fillna(0)

        result = df[df["Performance of Conversion Rate (%)"]>=0]
        result_Data = result.drop(columns=['Experiment Name', 'Store Listing', 'Experiment Type', 'Status', 'Audience',
                              'Performance (90% confidence interval)', 'Performance (90% confidence interval).1', 
                              'Performance (90% confidence interval).2', 'Performance (90% confidence interval).3'], 
                             inplace=False)
        
        if st.checkbox("Show Updated Dataframe After Performance of Conversion Rate Metrics:"):
           st.dataframe(result_Data)
        
        st.subheader("Filtered Data considered for Short Description Generation in order of Performance of Conversion Rates")
        data = result_Data[['Variant (Content)','Performance of Conversion Rate (%)']].reset_index()
        df_sorted = data.sort_values(by="Performance of Conversion Rate (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df_sorted.head(5))
        
        highest = df_sorted.iloc[0]["Variant (Content)"]         # Getting the highest Performance of Conversion Rates
        second_highest = df_sorted.iloc[1]["Variant (Content)"]  # Getting the 2nd-highest Performance of Conversion Rates
        third_highest = df_sorted.iloc[2]["Variant (Content)"]   # Getting the 3rd-highest Performance of Conversion Rates
    
        # Defining high-performing prompts based on the top Performance of Conversion Rate variants
        
        stoplist = ["#1", "Best of", "discount", "free for a limited time", "popular", 
            "Google", "WiFi", "hotspots"]
        # Define high-performing prompts based on the top conversion rate variants
        prompt1 = highest + "secure, global, reliable,"
        prompt2 = second_highest + "global, access"
        prompt3 = third_highest + "fast, access, phone"
        
        def filter_stoplist_words(text, stoplist):
            for word in stoplist:
                text = text.replace(word, "")  # Replacing each stoplist word with ""
            return text.strip()  # Removing leading/trailing whitespace

        def generate_text(prompt, max_length=25, temperature=0.75):
            input_ids = tokenizer.encode(prompt, return_tensors="pt")   # Encoding the prompt
            output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=temperature)    # Generate text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)   # Decoding the generated text
            return filter_stoplist_words(generated_text, stoplist)

        def generate_summary(text):
            summarizer = pipeline("summarization")
            summary = summarizer(text, max_length=21, min_length=10, do_sample=False)
            return summary[0]['summary_text']
            
        # Generate two new variants
        new_variant1 = generate_summary(generate_text(prompt1))
        new_variant2 = generate_summary(generate_text(prompt2))
        new_variant3 = generate_summary(generate_text(prompt3))
        
        print("Generated Variant 1:", new_variant1)
        print("Generated Variant 2:", new_variant2)
        print("Generated Variant 3:", new_variant3)
        
        st.title("Exploratory Data Analysis - A/B Tests")
        # WordCloud
        # Filtering out variants with high Performance of Conversion Rates (e.g., top 50% Performance of Conversion Rates)
        threshold = df['Performance of Conversion Rate (%)'].quantile(0.5)
        high_conversion_variants = df[df['Performance of Conversion Rate (%)'] >= threshold]
        # Combining all the high conversion variant content text
        text = " ".join(high_conversion_variants["Variant (Content)"])
        # Generating the word cloud
        wordcloud = WordCloud(width=1000, height=600, background_color='white', colormap='viridis', stopwords=set(stoplist)).generate(text)
        st.subheader("Word Cloud of High Performance of Conversion Rate Variant Words")
        
        # Plotting WordCloud
        plt.figure(figsize=(5, 3))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt) 
        
        # Bar Graph for Top Words in High Conversion Performance Variants
        # Tokenizing and counting words
        words = " ".join(high_conversion_variants["Variant (Content)"]).lower()
        words = re.findall(r'\b\w+\b', words)  # Extract words
        filtered_words = [word for word in words if word not in stoplist]  # Exclude stoplist words
        # Count most common words
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(15)
        labels, values = zip(*top_words)
        
        # Plotting Bar Graph
        plt.figure(figsize=(10, 7))
        plt.barh(labels, values, color='skyblue')
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.title("Top Words in High Conversion Variants")
        plt.gca().invert_yaxis()  # Inverting y-axis for better readability
        st.pyplot(plt)
        
        df['Start Date'] = pd.to_datetime(df['Start Date'])
        df['End Date'] = pd.to_datetime(df['End Date'])
        plt.figure(figsize=(12, 6))
        plt.plot(df['Start Date'], df['Performance of Conversion Rate (%)'], marker='o', linestyle='-', color='b', label='Performance (Start Date)')
        plt.plot(df['End Date'], df['Performance of Conversion Rate (%)'], marker='s', linestyle='--', color='orange', label='Performance (End Date)')
        plt.title('Performance with Different X-Axis Variables', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Performance %', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        
        st.title("Hypothesis statement Recommendation")
        st.subheader("AI Generated 1st Variant of Short Description")
        st.write(new_variant1)
        st.subheader("AI Generated 2nd Variant of Short Description")
        st.write(new_variant2)
        st.subheader("AI Generated 3rd Variant of Short Description")
        st.write(new_variant3)  
        st.write("\n")
    else:
        st.write("Please upload a file to proceed")
 
if st.checkbox("Short Description Generation - Metadata of App"):
    nlp = spacy.load('en_core_web_sm')
    banned_phrases = ["#1", "Best of", "discount", "free for a limited time", "popular", "WiFi"]
    
    def extract_key_phrases(text, max_phrases=5):
        doc = nlp(text)
        phrases = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() not in nlp.Defaults.stop_words]
        return phrases[:max_phrases]
    
    def generate_text(prompt, max_new_tokens=70, temperature=0.65,num_sequences=2):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")  # Encode the prompt
        output = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,  
            num_return_sequences=1, 
            temperature=temperature
        )  # Generating text
        #generated_text = tokenizer.decode(output[0], skip_special_tokens=True) 
        #return generated_text
        if output.size(0) < num_sequences:
           num_sequences = output.size(0) 
        generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(0,num_sequences)]
        return generated_texts

    def generate_summary(text):
        summarizer = pipeline("summarization")
        summary = summarizer(text, max_length=90, min_length=15, do_sample=False)
        return summary[0]['summary_text']
    
    def remove_banned_phrases(description, banned_phrases):
      for phrase in banned_phrases:
        description = description.replace(phrase, "")
      return description.strip()
  
    def truncate_text(text, max_tokens=100):
        input_ids = tokenizer.encode(text, return_tensors="pt")
        truncated_ids = input_ids[:, :max_tokens]  
        return tokenizer.decode(truncated_ids[0], skip_special_tokens=True)
    
    def truncate_to_80_characters(description, max_length=90):
        if len(description) <= max_length:
            return description
        truncated = description[:max_length].rsplit(' ', 1)[0]  # Avoiding to cut off mid-word
        return truncated + "..." if len(description) > max_length else truncated


    generated_texts = generate_text(truncate_text(metadata["long_description"]), num_sequences=2)
    short_descriptions = []

    for text in generated_texts:
        summary = generate_summary(text)
        cleaned_summary = remove_banned_phrases(summary, banned_phrases)
        truncated_summary = truncate_to_80_characters(cleaned_summary)
        short_descriptions.append(truncated_summary)
        
    st.dataframe(metadata_dataframe)
    st.header("Phrases from Metadata -> Key Phrases for Short Description Generation")
    st.write(extract_key_phrases(metadata["long_description"]))
    st.header("Generated Short Descriptions from Metadata of App:")
    for i, desc in enumerate(short_descriptions, 1):
        st.write(f"Description {i}: {desc}")

   
    

        
    
    
    
    
    
    
    
