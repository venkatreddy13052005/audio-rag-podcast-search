import os
import pickle
import faiss

import numpy as np
from sentence_transformers import SentenceTransformer
import load_dotenv
import openai
import streamlit as st
import time

# Load environment variables from a .env file
load_dotenv()

# --- Initialize OpenAI Client ---
try:
    client = openai.OpenAI()
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")
    st.stop()

# --- Streamlit App UI ---
st.set_page_config(page_title="Podcast Q&A with OpenAI", page_icon="🎙️")
st.title("🎙️ Podcast Q&A with OpenAI")
st.markdown("Ask a question about the podcast, and I'll find the most relevant segments to generate an answer.")

# --- Resource Loading with Caching ---
@st.cache_resource
def load_resources():
    """
    Loads the FAISS index, chunk metadata, and SentenceTransformer model.
    """
    try:
        st.write("Loading resources...")
        index = faiss.read_index("index/index.faiss")
        with open("index/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)  # metadata should contain "chunks": [...]
        model = SentenceTransformer("all-MiniLM-L6-v2")
        st.success("Resources loaded successfully!")
        return index, metadata, model
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

index, metadata, model = load_resources()

# --- Retrieval Function ---
def retrieve_chunks(query_embedding, k=5):
    """
    Searches the FAISS index for the top k most similar chunks.
    """
    D, I = index.search(np.array(query_embedding).astype("float32"), k=k)
    retrieved_chunks = [metadata["chunks"][i] for i in I[0]]
    return retrieved_chunks

# --- OpenAI Q&A Function ---
def query_openai(context, question):
    """
    Generates an answer using the retrieved context and the user's question.
    """
    prompt = f"""You are a helpful assistant answering questions based on podcast transcripts. 
Use only the provided context to answer the question. If the answer is not in the context,
state that you cannot find the answer in the provided information.

Context:
\"\"\"{context}\"\"\"

Question: {question}
Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for podcast Q&A."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        return f"An error occurred with the OpenAI API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Main App Logic ---
query = st.text_input("Ask your question here:", placeholder="e.g., What did the host say about the new study?")

if query:
    answer_placeholder = st.empty()
    
    with st.spinner("Searching for relevant podcast segments..."):
        query_embedding = model.encode([query])
        retrieved_chunks = retrieve_chunks(query_embedding, k=5)
        
        st.markdown("### 🔍 Retrieved Segments:")
        for i, chunk in enumerate(retrieved_chunks):
            if isinstance(chunk, dict) and 'text' in chunk:
                st.info(f"**Segment {i+1}**\n\n{chunk['text']}")
            else:
                st.info(f"**Segment {i+1}**\n\n{chunk}")
        
        # Build the context for OpenAI
        if any(isinstance(chunk, str) and chunk for chunk in retrieved_chunks):
            context = "\n\n".join(chunk for chunk in retrieved_chunks if isinstance(chunk, str) and chunk)
        elif any(isinstance(chunk, dict) and 'text' in chunk for chunk in retrieved_chunks):
            context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks if 'text' in chunk])
        else:
            context = ""
            st.warning("No valid text chunks were retrieved to form a context.")
    
    if context:
        with st.spinner("Generating answer with OpenAI..."):
            answer = query_openai(context, query)
        st.markdown("### 💬 Answer:")
        st.write(answer)
    else:
        st.warning("I'm sorry, I couldn't find any relevant information to answer your question.")



