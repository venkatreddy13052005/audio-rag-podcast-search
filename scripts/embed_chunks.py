import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "..", "chunks")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")

os.makedirs(INDEX_DIR, exist_ok=True)

print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

all_chunks_text = []
all_embeddings = []

print(f"Processing chunk files in {CHUNKS_DIR}...")
chunk_files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".txt")]

if not chunk_files:
    print("❌ Error: No .txt chunk files found.")
    exit()

for filename in chunk_files:
    path = os.path.join(CHUNKS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        chunk_text = f.read().strip()
        if chunk_text:
            all_chunks_text.append(chunk_text)
            embedding = model.encode(chunk_text)
            all_embeddings.append(embedding)

print(f"✅ Processed {len(all_chunks_text)} chunks.")

if not all_embeddings:
    print("❌ Error: No valid chunks found.")
    exit()

# Build FAISS index
print("Converting embeddings to a NumPy array...")
embeddings_array = np.array(all_embeddings).astype("float32")

print("Creating and populating FAISS index...")
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# Save index
index_path = os.path.join(INDEX_DIR, "index.faiss")
print(f"Saving FAISS index to {index_path}...")
faiss.write_index(index, index_path)

# Save metadata as a dictionary
metadata = {
    "chunks": all_chunks_text
}
metadata_path = os.path.join(INDEX_DIR, "metadata.pkl")
print(f"Saving metadata to {metadata_path}...")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

print(f"\n✅ All done! Total chunks indexed: {len(all_chunks_text)}")
