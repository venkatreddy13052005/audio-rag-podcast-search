import whisper
import os
import time
from pydub import AudioSegment

AUDIO_DIR = "../audio"
TRANSCRIPT_DIR = "../transcripts"
TRIMMED_DIR = "../trimmed_audio"
# Changed the max duration from 2 minutes (120 seconds) to 5 minutes (300 seconds)
MAX_DURATION_SECONDS = 1200  # 5 minutes

# Load Whisper model
# Change to "small", "medium", or "large" if you need better accuracy
model = whisper.load_model("base")

# Ensure output directories exist
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(TRIMMED_DIR, exist_ok=True)

# Supported audio formats
SUPPORTED_FORMATS = (".mp3", ".wav", ".m4a", ".flac")

for filename in os.listdir(AUDIO_DIR):
    if filename.lower().endswith(SUPPORTED_FORMATS):
        audio_path = os.path.join(AUDIO_DIR, filename)
        trimmed_path = os.path.join(TRIMMED_DIR, f"trimmed_{filename}")
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"{os.path.splitext(filename)[0]}.txt")

        # Skip if already transcribed
        if os.path.exists(transcript_path):
            print(f"⏩ Skipping (already transcribed): {filename}")
            continue

        try:
            print(f"\n✂️ Trimming (first 5 minutes): {filename}")
            audio = AudioSegment.from_file(audio_path)
            trimmed_audio = audio[:MAX_DURATION_SECONDS * 1000]  # convert to milliseconds
            trimmed_audio.export(trimmed_path, format="mp3")

            print(f"🎙️ Transcribing trimmed audio: trimmed_{filename}")
            start_time = time.time()
            result = model.transcribe(trimmed_path)
            duration = time.time() - start_time

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            print(f"✅ Done in {duration:.2f} seconds → {transcript_path}")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {str(e)}")
