"""
Text processing module for chunking transcribed text and generating embeddings.
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import ssl

# Download NLTK data with SSL workaround
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('punkt', quiet=True)
except:
    pass

class TextProcessor:
    """Handles text chunking and embedding generation for podcast transcripts."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the text processor.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        try:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded successfully!")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def chunk_transcript_semantic(self, segments: List[Dict], 
                                chunk_size_seconds: float = 60.0,
                                overlap_seconds: float = 10.0) -> List[Dict]:
        """
        Chunk transcript segments semantically with time-based windows.
        
        Args:
            segments: List of transcript segments with timestamps
            chunk_size_seconds: Target chunk size in seconds
            overlap_seconds: Overlap between chunks in seconds
            
        Returns:
            List of text chunks with metadata
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk_text = []
        current_chunk_start = segments[0]["start_time"]
        current_chunk_segments = []
        
        for segment in segments:
            segment_start = segment["start_time"]
            segment_end = segment["end_time"]
            
            # Check if we should start a new chunk
            if (segment_start - current_chunk_start) >= chunk_size_seconds and current_chunk_text:
                # Create chunk from accumulated segments
                chunk = self._create_chunk(current_chunk_segments, current_chunk_start)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start_time = current_chunk_start + chunk_size_seconds - overlap_seconds
                current_chunk_segments = [s for s in current_chunk_segments 
                                        if s["start_time"] >= overlap_start_time]
                current_chunk_text = [s["text"] for s in current_chunk_segments]
                current_chunk_start = overlap_start_time if current_chunk_segments else segment_start
            
            # Add current segment to chunk
            current_chunk_segments.append(segment)
            current_chunk_text.append(segment["text"])
        
        # Add final chunk
        if current_chunk_segments:
            chunk = self._create_chunk(current_chunk_segments, current_chunk_start)
            chunks.append(chunk)
        
        return chunks
    
    def chunk_transcript_sentence_based(self, segments: List[Dict],
                                      max_sentences: int = 10,
                                      overlap_sentences: int = 2) -> List[Dict]:
        """
        Chunk transcript based on sentence boundaries.
        
        Args:
            segments: List of transcript segments with timestamps
            max_sentences: Maximum sentences per chunk
            overlap_sentences: Number of overlapping sentences between chunks
            
        Returns:
            List of text chunks with metadata
        """
        if not segments:
            return []
        
        # Combine all text and split into sentences
        full_text = " ".join([seg["text"] for seg in segments])
        sentences = sent_tokenize(full_text)
        
        # Map sentences back to segments for timing information
        sentence_segments = self._map_sentences_to_segments(sentences, segments)
        
        chunks = []
        for i in range(0, len(sentence_segments), max_sentences - overlap_sentences):
            chunk_sentences = sentence_segments[i:i + max_sentences]
            if chunk_sentences:
                chunk = self._create_chunk_from_sentences(chunk_sentences)
                chunks.append(chunk)
        
        return chunks
    
    def chunk_transcript_topic_based(self, segments: List[Dict],
                                   similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Chunk transcript based on topic similarity (experimental).
        
        Args:
            segments: List of transcript segments with timestamps
            similarity_threshold: Threshold for topic similarity
            
        Returns:
            List of text chunks with metadata
        """
        if not segments or len(segments) < 2:
            return [self._create_chunk(segments, segments[0]["start_time"])] if segments else []
        
        # Generate embeddings for each segment
        segment_texts = [seg["text"] for seg in segments]
        embeddings = self.embedding_model.encode(segment_texts)
        
        # Find topic boundaries based on embedding similarity
        chunks = []
        current_chunk_segments = [segments[0]]
        
        for i in range(1, len(segments)):
            # Calculate similarity with previous segment
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            if similarity < similarity_threshold:
                # Topic change detected, create new chunk
                chunk = self._create_chunk(current_chunk_segments, current_chunk_segments[0]["start_time"])
                chunks.append(chunk)
                current_chunk_segments = [segments[i]]
            else:
                current_chunk_segments.append(segments[i])
        
        # Add final chunk
        if current_chunk_segments:
            chunk = self._create_chunk(current_chunk_segments, current_chunk_segments[0]["start_time"])
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, segments: List[Dict], start_time: float) -> Dict:
        """Create a chunk from a list of segments."""
        if not segments:
            return {}
        
        text = " ".join([seg["text"] for seg in segments]).strip()
        end_time = segments[-1]["end_time"]
        
        # Get metadata from first segment
        first_segment = segments[0]
        
        # Collect unique speakers in this chunk
        speakers = list(set([seg["speaker"] for seg in segments if seg["speaker"] != "Unknown"]))
        
        chunk = {
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "episode_title": first_segment["episode_title"],
            "audio_file": first_segment["audio_file"],
            "speakers": speakers,
            "segment_count": len(segments),
            "chunk_id": f"{first_segment['episode_title']}_{start_time:.1f}_{end_time:.1f}",
            "original_segments": segments
        }
        
        return chunk
    
    def _create_chunk_from_sentences(self, sentence_segments: List[Dict]) -> Dict:
        """Create a chunk from sentence segments."""
        if not sentence_segments:
            return {}
        
        text = " ".join([sent["text"] for sent in sentence_segments])
        start_time = sentence_segments[0]["start_time"]
        end_time = sentence_segments[-1]["end_time"]
        
        # Extract metadata
        episode_title = sentence_segments[0]["episode_title"]
        audio_file = sentence_segments[0]["audio_file"]
        speakers = list(set([sent["speaker"] for sent in sentence_segments if sent["speaker"] != "Unknown"]))
        
        chunk = {
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "episode_title": episode_title,
            "audio_file": audio_file,
            "speakers": speakers,
            "sentence_count": len(sentence_segments),
            "chunk_id": f"{episode_title}_{start_time:.1f}_{end_time:.1f}",
            "original_sentences": sentence_segments
        }
        
        return chunk
    
    def _map_sentences_to_segments(self, sentences: List[str], segments: List[Dict]) -> List[Dict]:
        """Map sentences back to original segments for timing information."""
        sentence_segments = []
        segment_idx = 0
        char_position = 0
        
        for sentence in sentences:
            # Find which segment(s) this sentence belongs to
            sentence_start_char = char_position
            sentence_end_char = char_position + len(sentence)
            
            # Find corresponding segments
            while segment_idx < len(segments):
                segment = segments[segment_idx]
                segment_text = segment["text"]
                
                # Check if sentence overlaps with this segment
                if sentence.strip() in segment_text or any(word in segment_text for word in sentence.split()[:3]):
                    sentence_segment = {
                        "text": sentence.strip(),
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "speaker": segment["speaker"],
                        "episode_title": segment["episode_title"],
                        "audio_file": segment["audio_file"]
                    }
                    sentence_segments.append(sentence_segment)
                    break
                
                segment_idx += 1
            
            char_position = sentence_end_char + 1
        
        return sentence_segments
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def enhance_text_for_search(self, text: str) -> str:
        """
        Enhance text for better search capabilities.
        
        Args:
            text: Original text
            
        Returns:
            Enhanced text with additional context
        """
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add period if missing (for better sentence segmentation)
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract key terms from text (simple frequency-based approach).
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 
                     'have', 'has', 'had', 'will', 'would', 'could', 'should',
                     'this', 'that', 'these', 'those', 'they', 'them', 'their'}
        
        words = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top_k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]