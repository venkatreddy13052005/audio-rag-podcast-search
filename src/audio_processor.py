"""
Audio processing module for converting audio to text with timestamps and speaker identification.
"""

import os
import io
import tempfile
import numpy as np
import librosa
import soundfile as sf
import whisper
import torch
import warnings
from pydub import AudioSegment
from scipy.io import wavfile
from typing import List, Dict, Tuple, Optional
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("whisper").setLevel(logging.WARNING)

class AudioProcessor:
    """Handles audio preprocessing, speech-to-text, and speaker diarization."""
    
    def __init__(self, whisper_model_size: str = "base"):
        """
        Initialize the audio processor.
        
        Args:
            whisper_model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.whisper_model_size = whisper_model_size
        self.whisper_model = None
        self.diarization_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and speaker diarization models."""
        try:
            # Load Whisper model
            print(f"Loading Whisper model ({self.whisper_model_size})...")
            self.whisper_model = whisper.load_model(self.whisper_model_size)
            print("Whisper model loaded successfully!")
            
            # Load speaker diarization model (pyannote.audio)
            try:
                from pyannote.audio import Pipeline
                print("Loading speaker diarization model...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=False  # Set to your HF token if needed
                )
                print("Speaker diarization model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load speaker diarization model: {e}")
                print("Speaker identification will be disabled.")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def preprocess_audio(self, audio_file_path: str) -> str:
        """
        Preprocess audio file: convert to WAV, normalize, and reduce noise.
        
        Args:
            audio_file_path: Path to the input audio file
            
        Returns:
            Path to the preprocessed audio file
        """
        try:
            # Load audio file using pydub (supports many formats)
            audio = AudioSegment.from_file(audio_file_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate to 16kHz (optimal for Whisper)
            audio = audio.set_frame_rate(16000)
            
            # Normalize audio
            audio = audio.normalize()
            
            # Apply basic noise reduction (simple high-pass filter)
            # Remove frequencies below 80Hz (typical for removing low-frequency noise)
            audio = audio.high_pass_filter(80)
            
            # Create temporary file for processed audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio.export(temp_file.name, format="wav")
                return temp_file.name
                
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return audio_file_path  # Return original if preprocessing fails
    
    def transcribe_with_timestamps(self, audio_file_path: str) -> Dict:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription results with timestamps
        """
        try:
            # Preprocess audio
            processed_audio_path = self.preprocess_audio(audio_file_path)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                processed_audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            # Clean up temporary file if it was created
            if processed_audio_path != audio_file_path:
                os.unlink(processed_audio_path)
            
            return result
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise
    
    def identify_speakers(self, audio_file_path: str) -> Optional[Dict]:
        """
        Perform speaker diarization to identify different speakers.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing speaker segments or None if diarization fails
        """
        if self.diarization_pipeline is None:
            return None
            
        try:
            # Run speaker diarization
            diarization = self.diarization_pipeline(audio_file_path)
            
            # Convert to our format
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            return {"segments": speaker_segments}
            
        except Exception as e:
            print(f"Warning: Speaker diarization failed: {e}")
            return None
    
    def process_podcast_episode(self, audio_file_path: str, episode_title: str = None) -> Dict:
        """
        Complete processing pipeline for a podcast episode.
        
        Args:
            audio_file_path: Path to the podcast audio file
            episode_title: Optional title for the episode
            
        Returns:
            Dictionary containing all processing results
        """
        print(f"Processing podcast episode: {episode_title or 'Unknown'}")
        
        # Transcribe with timestamps
        print("Transcribing audio...")
        transcription_result = self.transcribe_with_timestamps(audio_file_path)
        
        # Identify speakers
        print("Identifying speakers...")
        speaker_result = self.identify_speakers(audio_file_path)
        
        # Combine results
        result = {
            "episode_title": episode_title or "Unknown Episode",
            "audio_file": audio_file_path,
            "transcription": transcription_result,
            "speakers": speaker_result,
            "duration": transcription_result.get("segments", [{}])[-1].get("end", 0) if transcription_result.get("segments") else 0
        }
        
        print(f"Processing complete! Duration: {result['duration']:.1f} seconds")
        return result
    
    def extract_segments_with_metadata(self, processing_result: Dict) -> List[Dict]:
        """
        Extract text segments with timestamps and speaker information.
        
        Args:
            processing_result: Result from process_podcast_episode
            
        Returns:
            List of text segments with metadata
        """
        segments = []
        transcription = processing_result.get("transcription", {})
        speakers = processing_result.get("speakers", {})
        
        # Create speaker mapping based on timestamps
        speaker_map = {}
        if speakers and "segments" in speakers:
            for segment in speakers["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                speaker_id = segment["speaker"]
                
                # Map time ranges to speakers
                for t in range(int(start_time), int(end_time) + 1):
                    speaker_map[t] = speaker_id
        
        # Process transcription segments
        for segment in transcription.get("segments", []):
            text = segment["text"].strip()
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Find speaker for this segment
            segment_middle = (start_time + end_time) / 2
            speaker_id = speaker_map.get(int(segment_middle), "Unknown")
            
            segment_data = {
                "text": text,
                "start_time": start_time,
                "end_time": end_time,
                "speaker": speaker_id,
                "episode_title": processing_result["episode_title"],
                "audio_file": processing_result["audio_file"],
                "segment_id": f"{processing_result['episode_title']}_{start_time:.1f}_{end_time:.1f}"
            }
            
            segments.append(segment_data)
        
        return segments

# Utility functions for audio handling
def get_audio_duration(file_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0  # Convert ms to seconds
    except:
        return 0.0

def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"