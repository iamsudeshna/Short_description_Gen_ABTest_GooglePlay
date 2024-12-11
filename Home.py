"""Created on Wed Oct 16 23:52:49 2024
@author: sudeshna.a.kundu
"""
import streamlit as st
st.set_page_config(
    page_title = 'Home',
    page_icon="🏠",
    layout = 'wide')

st.title("About the Project")
st.write("We are analysing the app Instabridge for our usecase. Its an app which locates and provides free wifi. We are trying to analyse from the past ratings and reviews of Instabridge itself, and trying interpret the positive and negative sentiments of the reviews")
st.write("We are also trying to generate better short descriptions from the positive sentiments of reviews and Past AB Tests Data for short descriptions")
st.write("We are also having few competitors of Instabridge, which are, Wifi Map, WifiMan. From the reviews of these competitor apps we will analyse the sentiments and infer how and why these are better or worse than Instabridge and how they are in competition with Instabridge")
# Create a dropdown menu
apps = ["InstaBridge"]
Competitors = ["Wifi Map", "WifiMan"]
language = ["English"]
selected_apps = st.selectbox("Choose the app for analysis:", apps)
selected_competitor = st.selectbox("Choose the competitor app for analysis:", Competitors)
selected_language = st.selectbox("Choose the language for analysis and Content generation:", language)


