import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os

# Set page config FIRST
st.set_page_config(page_title="📚 BookBot", page_icon="🤖", layout="centered")

# --- Load Data + Embeddings ---
@st.cache_resource
def load_data():
    with open('book_dataset_with_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_data()
book_embeddings = np.array(data['books']['embeddings'])
book_texts = data['books']['texts']
book_titles = data['books']['titles']
book_authors = data['books']['authors']

# --- Embed User Query ---
@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = get_embedder()

# --- Helper: Query LLaMA ---
def query_llama(prompt):
    client = openai.OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )
    response = client.chat.completions.create(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content.strip()

# --- Similarity Search ---
def recommend_books(user_input, top_n=5):
    query_embedding = embedder.encode([user_input])
    similarities = cosine_similarity(query_embedding, book_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n + 1]
    return top_indices, similarities[top_indices]

# --- Streamlit UI ---
st.title("📚 Ask BookBot for Recommendations")
st.markdown("Just type what you're in the mood to read and BookBot will recommend something!")

# Conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.chat_input("Ask for a book recommendation...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Find matches
    top_indices, scores = recommend_books(user_input)
    best_match = top_indices[0]
    similar_books = top_indices[1:]

    # Format recommendations for prompt
    prompt = f"""
You are a book recommender AI. The user asked: "{user_input}".

The best match is: "{book_titles[best_match]}" by {book_authors[best_match]}.

Here are 5 similar books:
{chr(10).join([f'- "{book_titles[i]}" by {book_authors[i]}' for i in similar_books])}

Write a short and friendly response. If the user’s query is unclear, feel free to ask them a clarifying question.
"""
    # Get response from LLaMA
    reply = query_llama(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
