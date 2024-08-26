import streamlit as st
import numpy as np
from langchain import PromptTemplate
import google.generativeai as genai
import faiss
import os
import pickle as pickle
from langchain_community.embeddings import HuggingFaceEmbeddings


# Function definitions
def load_faiss_index(faiss_index_file, chunks_file):
    # Load FAISS index from file
    faiss_index = faiss.read_index(faiss_index_file)
    
    # Load the chunks from a pickle file
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)
    
    return faiss_index, chunks

def embed_question(question):
    embeddings = HuggingFaceEmbeddings()
    return embeddings.embed_query(question)

def get_context (question, faiss_index, chunks):
    query_embedding = embed_question(question)
    _, indices = search_faiss_index(faiss_index, query_embedding, k=5)
    similar_chunks = get_similar_chunks(indices, chunks)
    context = "\n\n".join([c.page_content for c in similar_chunks])
    return context

def get_prompt_template(context, question):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the following context YOU ARE STRICTLY REQUIRED TO ANSWER WITHIN THE DOMAIN OF THE CONTEXT:\n\n{context}\n\nAnswer the question:\n\n{question}\n\n"
    )
    formatted_prompt = prompt_template.template.format(context=context, question=question)
    return formatted_prompt
def search_faiss_index(index, query_embedding, k=5):
    query_embedding_array = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_embedding_array, k)
    return distances, indices

# Retrieve the most similar chunks
def get_similar_chunks(indices, chunks):
    return [chunks[i] for i in indices[0]]

# Initialize Streamlit app
st.title("PDF Question Answering System")

# Paths to the FAISS index and chunk files
faiss_index_file = "faiss_index_file.index" 
chunks_file = "chunks_file.pkl"  

if os.path.exists(faiss_index_file) and os.path.exists(chunks_file):
    faiss_index, chunks = load_faiss_index(faiss_index_file, chunks_file)
    
    # User input for the question
    question = st.text_input("Ask a question related to the indexed PDF content")

    if question:
        # Get context from the FAISS index
        context = get_context(question, faiss_index, chunks)
        
        # Generate the formatted prompt
        formatted_prompt = get_prompt_template(context, question)
        api_key = "Enter your own API KEY"
        # Get the response from the Gemini model
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(formatted_prompt)
        # Display the response
        st.write("### Response:")
        st.write(response.text)
else:
    st.error("FAISS index file or chunks file not found. Please ensure 'faiss_index.index' and 'chunks.pkl' are in the same directory.")
