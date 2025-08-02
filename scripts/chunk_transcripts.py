import os
import json
import nltk
import pickle
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

TRANSCRIPTS_DIR = "transcripts"
OUTPUT_PATH = "index/metadata.pkl"
CHUNK_SIZE = 1000  # chars per chunk
CHUNK_OVERLAP = 200  # chars overlap

def chunk_with_timestamps(transcript, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    buffer = ""
    start_time = None

    for segment in transcript:
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]

        sentences = sent_tokenize(text)

        for sentence in sentences:
            if not buffer:
                start_time = start

            if len(buffer) + len(sentence) <= chunk_size:
                buffer += " " + sentence
            else:
                chunks.append({
                    "chunk": buffer.strip(),
                    "start": start_time,
                    "end": end
                })
                buffer = buffer[-overlap:] + " " + sentence
                start_time = start  # new chunk starts here

    if buffer.strip():
        chunks.append({
            "chunk": buffer.strip(),
            "start": start_time,
            "end": transcript[-1]["end"]
        })

    return chunks

def main():
    os.makedirs("index", exist_ok=True)
    all_chunks = []

    for file in os.listdir(TRANSCRIPTS_DIR):
        if file.endswith(".json"):
            path = os.path.join(TRANSCRIPTS_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                transcript = json.load(f)

            episode_chunks = chunk_with_timestamps(transcript)

            for chunk in episode_chunks:
                chunk["episode"] = file  # track source
            all_chunks.extend(episode_chunks)

            print(f"✅ {file}: {len(episode_chunks)} chunks")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n📦 metadata.pkl saved with {len(all_chunks)} total chunks.")

if __name__ == "__main__":
    main()
