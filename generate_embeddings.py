import pandas as pd
import os
import numpy as np
import csv
import pickle
from sentence_transformers import SentenceTransformer
import kagglehub

# --- Download dataset from Kaggle ---
print("📥 Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")
print(f"✅ Dataset downloaded to: {path}")



# --- Load CSV Files ---
books_path = os.path.join(path, "Books.csv")
users_path = os.path.join(path, "Users.csv")
ratings_path = os.path.join(path, "Ratings.csv")

books_df = pd.read_csv(
    books_path,
    sep=',',  # ← changed from semicolon to comma
    encoding='latin-1',
    header=0,
    on_bad_lines='skip'  # still useful for broken rows
)
users_df = pd.read_csv(
    users_path,
    sep=';',
    encoding='latin-1',
    quotechar='"',
    on_bad_lines='skip'
)
ratings_df = pd.read_csv(
    ratings_path,
    sep=';',
    encoding='latin-1',
    quotechar='"',
    on_bad_lines='skip'
)

# Clean up column names just in case
books_df.columns = books_df.columns.str.strip()

users_df.columns = users_df.columns.str.strip()
ratings_df.columns = ratings_df.columns.str.strip()

# Show header preview to confirm
print("✅ Books CSV header preview:")
print(books_df.columns.tolist())

# Select relevant columns
books_df = books_df[['ISBN', 'Book-Title', 'Book-Author', 'Publisher']]

books_df = books_df.dropna().drop_duplicates(subset='ISBN')

# --- Combine text fields for embedding ---
books_df['text'] = books_df.apply(
    lambda x: f"{x['Book-Title']} by {x['Book-Author']} - Publisher: {x['Publisher']}", axis=1
)

# --- Load SentenceTransformer ---
print("🤖 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Swap with local model if desired

# --- Generate Embeddings ---
print("🧠 Generating embeddings...")
embeddings = model.encode(books_df['text'].tolist(), show_progress_bar=True)

# --- Save everything as pickle ---
output_data = {
    'books': {
        'ISBNs': books_df['ISBN'].tolist(),
        'titles': books_df['Book-Title'].tolist(),
        'authors': books_df['Book-Author'].tolist(),
        'texts': books_df['text'].tolist(),
        'embeddings': embeddings
    },
    'users': users_df.to_dict(orient='records'),
    'ratings': ratings_df.to_dict(orient='records')
}

with open('book_dataset_with_embeddings.pkl', 'wb') as f:
    pickle.dump(output_data, f)

print("✅ Dataset + Embeddings saved to 'book_dataset_with_embeddings.pkl'")
