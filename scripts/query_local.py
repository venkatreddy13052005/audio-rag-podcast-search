import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "..", "index", "index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "..", "index", "metadata.pkl")

print("🔁 Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

chunks = metadata["chunks"]
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Query ---
query = input("\n🔍 Enter your query: ").strip()
if not query:
    print("⚠️ Empty query. Exiting.")
    exit()

print("📡 Embedding query...")
query_embedding = model.encode(query).astype("float32")

# --- Search ---
k = 3  # number of results
print(f"🔎 Searching top {k} matches...")
distances, indices = index.search(np.array([query_embedding]), k)

# --- Show Results ---
print("\n📌 Top Results:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"\n[Match #{rank}] (Distance: {dist:.4f})")
    print(chunks[idx])
