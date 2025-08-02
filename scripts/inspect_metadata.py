import os
import pickle

# Construct absolute path to metadata.pkl
script_dir = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, "..", "index", "metadata.pkl")

# Load metadata
if not os.path.exists(metadata_path):
    print(f"❌ metadata.pkl not found at: {metadata_path}")
    exit()

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# Check and display contents
if isinstance(metadata, list):
    print(f"\n✅ Loaded {len(metadata)} metadata entries. Previewing first 3:\n")
    for i, entry in enumerate(metadata[:3]):
        print(f"[ENTRY {i+1}]")
        for key, value in entry.items():
            print(f"{key}: {value}")
        print()
elif isinstance(metadata, dict):
    print("\n✅ KEYS in metadata:", list(metadata.keys()))
    chunks = metadata.get("chunks", [])
    if not chunks:
        print("⚠️ No chunks found.")
    else:
        print(f"🔍 Found {len(chunks)} chunks. Previewing first 3:\n")
        for i, chunk in enumerate(chunks[:3]):
            print(f"[CHUNK {i+1}]\n{chunk}\n")
else:
    print(f"⚠️ Unexpected metadata format: {type(metadata)}")
