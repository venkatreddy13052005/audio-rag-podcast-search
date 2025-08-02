import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Paths
METADATA_PATH = "../index/metadata.pkl"
FAISS_INDEX_PATH = "../index/podcast_index.index"

# Load and validate metadata
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)  # List of dicts with 'chunk', 'episode', 'start', 'end'

# Remove invalid or empty entries
metadata = [entry for entry in metadata if "chunk" in entry and entry["chunk"].strip()]
if not metadata:
    raise ValueError("No valid chunk data found in metadata.")

# Optional: Remove duplicate chunks
unique_metadata = []
seen_chunks = set()
for entry in metadata:
    chunk_text = entry["chunk"]
    if chunk_text not in seen_chunks:
        seen_chunks.add(chunk_text)
        unique_metadata.append(entry)

metadata = unique_metadata
chunks = [entry["chunk"] for entry in metadata]

print("✅ Loaded", len(chunks), "unique chunks for indexing.")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)

print("✅ FAISS index built and saved to", FAISS_INDEX_PATH)


# --- Semantic Search Function ---
def search(query: str, k: int = 5) -> List[Tuple[str, str, float, float, float]]:
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i == -1 or np.isnan(dist) or dist > 1e6:
            continue
        entry = metadata[i]
        chunk = entry["chunk"]
        source = entry.get("episode", "unknown")
        start = entry.get("start", 0.0)
        end = entry.get("end", 0.0)
        similarity = 1 / (1 + dist)  # Optional: Convert L2 distance to similarity
        results.append((source, chunk, similarity, start, end))

    return results


# --- CLI for Testing ---
if __name__ == "__main__":
    while True:
        query = input("\n🔍 Enter your search query (or 'exit'): ")
        if query.lower() == "exit":
            break

        top_k = 5
        results = search(query, k=top_k)

        if not results:
            print("⚠️ No results found.")
            continue

        print("\n📌 Top Results:")
        for idx, (source, chunk, score, start, end) in enumerate(results, 1):
            print(f"\n--- Result {idx} ---")
            print(f"🎧 Episode: {source}")
            print(f"⏱️ Time: {start:.2f} - {end:.2f} seconds")
            print(f"📄 Chunk: {chunk[:300]}...")
            print(f"🧠 Similarity Score: {score:.4f}")
