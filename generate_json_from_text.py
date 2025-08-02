import json
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

text = """I did a little experiment with a homeless person, not like on them, it's not like electrodes. With them voluntarily helped me. Because the whole idea of giving, right? You've all walked down the street and you've all seen someone begging and you either have or haven't thrown a few pennies in their cup. When you do, you feel good. You bought that feeling. That is a legitimate commercial transaction..."""  # full text here

sentences = sent_tokenize(text)

json_output = []
start = 0.0
duration_per_sentence = 4.5  # fake timestamps for testing

for i, sentence in enumerate(sentences):
    end = start + duration_per_sentence
    json_output.append({
        "start": round(start, 2),
        "end": round(end, 2),
        "text": sentence.strip()
    })
    start = end

with open("transcripts/sample1.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2)

print("✅ Saved transcript to transcripts/sample1.json")
