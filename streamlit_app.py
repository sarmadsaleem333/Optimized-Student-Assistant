import streamlit as st
import google.generativeai as genai
from langchain import PromptTemplate, LLMChain
from functions import DocumentQA
from langchain_community.document_loaders import PyPDFLoader
import os
# Show title and description.
st.title("📄 Document Question Answering with Gemini")
st.write(
    "Upload a document below and ask a question about it. Our model will answer it."
)

# Input field for API Key
api_key = st.text_input("API Key", type="password")
if not api_key:
    st.info("Please add your API key to continue.", icon="🗝️")
else:
    # Configure the Gemini API client with the provided API key
    genai.configure(api_key=api_key)

    # File uploader for the user to upload a document (PDF)
    uploaded_file = st.file_uploader("Upload a document (.pdf)", type=("pdf"))
    
    # Text area for the user to input a question about the document
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    # If both a file is uploaded and a question is asked
    if uploaded_file and question:
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Generate a response using the Gemini model
        response = model.generate_content(question)

        # Display the response in the Streamlit app
        st.write("### Answer:")
        st.write(response.text)