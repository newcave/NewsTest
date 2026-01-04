import streamlit as st
import pandas as pd
from google import genai

st.set_page_config(page_title="KIHS Online Data Analyzer (Demo)", layout="wide")

st.title("Korea Institute of Water Resources Research (KIHS)")
st.subheader("Monthly Related Online Data Analyzer (Demo Version)")

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

prompt = st.text_area(
    "Test prompt",
    "Summarize recent water resources research trends in Korea."
)

if st.button("Run Gemini"):
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    st.markdown(res.text)
