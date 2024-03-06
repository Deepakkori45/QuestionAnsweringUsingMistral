# import streamlit as st
# import textwrap

# # !pip install -q -U google-generativeai
# import google.generativeai as genai
# # from google.colab import userdata
# from IPython.display import Markdown

# def to_markdown(text):
#     text = text.replace('•', '  *')
#     return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# genai.configure(api_key="AIzaSyC5jVGT9OHx4soEsliU60ByZsieobJPRms")

# st.write("""
# # Content Generation App

# This app generates content using Google GenerativeAI.
# """)

# prompt = st.text_input('Enter your prompt:')
# if prompt:
#     model = genai.GenerativeModel('gemini-pro')
#     response = model.generate_content(prompt)
#     st.markdown(to_markdown(response.text))

import streamlit as st
import textwrap
import google.generativeai as genai

# Function to convert text to Markdown format
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# Configure Google GenerativeAI with your API key
genai.configure(api_key="AIzaSyC5jVGT9OHx4soEsliU60ByZsieobJPRms")

# Streamlit app title and description
st.write("""
# Question Answering System

This app generates content using Fine-Tuned model of Mistral 7b Model.
""")

# Text input for user prompt
prompt = st.text_input('Enter your prompt:')

# If user enters a prompt, generate content and display it
if prompt:
    # Instantiate Generative Model
    model = genai.GenerativeModel('gemini-pro')
    
    # Generate content using the prompt
    response = model.generate_content(prompt)
    
    # Display generated content
    st.write(to_markdown(response.text))
