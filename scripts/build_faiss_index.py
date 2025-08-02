import os
import pickle
import faiss
import numpy as np

# Load embeddings
print("📦 Loading embeddings: Simon Sinek । 30 Minutes for the NEXT 30 Years of Your LIFE_chunks_embeddings.pkl")
with open("../embeddings/Simon Sinek । 30 Minutes for the NEXT 30 Years of Your LIFE_chunks_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print("✅ Available keys in pickle:", list(data.keys()))

embeddings = data.get("embeddings")
metadata = data.get("metadata", [{} for _ in range(len(embeddings))])  # default to empty dicts if missing

# Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# Save FAISS index
os.makedirs("../index", exist_ok=True)
faiss.write_index(index, "../index/index.faiss")

# Save metadata
with open("../index/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAISS index and metadata saved in /index")
