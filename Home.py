"""Created on Wed Oct 16 23:52:49 2024
@author: sudeshna.a.kundu
"""
import streamlit as st
st.set_page_config(
    page_title = 'Home',
    page_icon="🏠",
    layout = 'wide')

st.title("About the Project")
st.write("We are analysing an App/Game for our usecase. Its an app which locates and provides free wifi. We are trying to analyse from the past ratings and reviews of the target app itself, and trying interpret the positive and negative sentiments of the reviews")
st.write("We are also trying to generate better short descriptions from the positive sentiments of reviews and Past AB Tests Data for short descriptions")
st.write("We are also having few competitors of our target App/Game. From the reviews of these competitor apps we will analyse the sentiments and infer how and why these are better or worse than our App and how they are in competition with the App/Game")
# Create a dropdown menu
apps = ["App_InstaBridge-123#"]
Competitors = ["Wifi Map - 101#", "WifiMan - 102#"]
language = ["English"]
selected_apps = st.selectbox("Choose the app for analysis:", apps)
selected_competitor = st.selectbox("Choose the competitor app for analysis:", Competitors)
selected_language = st.selectbox("Choose the language for analysis and Content generation:", language)


