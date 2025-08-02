"""
Vector store module for indexing and retrieving podcast transcript chunks.
"""

import os
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pickle


class PodcastVectorStore:
    """Vector store for podcast transcript chunks using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "podcast_transcripts"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Create or get collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                print(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Podcast transcript chunks with timestamps"}
                )
                print(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Add text chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of text chunks with metadata
            embeddings: Precomputed embeddings for the chunks
        """
        if not chunks or len(chunks) == 0:
            print("No chunks to add.")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings_list = []
            
            for i, chunk in enumerate(chunks):
                # Generate unique ID
                chunk_id = chunk.get("chunk_id", f"chunk_{uuid.uuid4()}")
                ids.append(chunk_id)
                
                # Document text
                documents.append(chunk["text"])
                
                # Metadata (ChromaDB requires JSON-serializable values)
                metadata = {
                    "episode_title": chunk.get("episode_title", "Unknown"),
                    "start_time": float(chunk.get("start_time", 0)),
                    "end_time": float(chunk.get("end_time", 0)),
                    "duration": float(chunk.get("duration", 0)),
                    "speakers": json.dumps(chunk.get("speakers", [])),  # Serialize list
                    "audio_file": chunk.get("audio_file", ""),
                    "segment_count": int(chunk.get("segment_count", 0)),
                    "timestamp_formatted": self._format_timestamp(chunk.get("start_time", 0))
                }
                metadatas.append(metadata)
                
                # Embedding
                embeddings_list.append(embeddings[i].tolist())
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            
            print(f"Added {len(chunks)} chunks to vector store.")
            
        except Exception as e:
            print(f"Error adding chunks to vector store: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, n_results: int = 10, 
               episode_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            episode_filter: Optional episode title to filter by
            
        Returns:
            List of search results with metadata
        """
        try:
            # Prepare query
            query_embeddings = [query_embedding.tolist()]
            
            # Build where clause for filtering
            where_clause = None
            if episode_filter:
                where_clause = {"episode_title": episode_filter}
            
            # Search
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                    "distance": results["distances"][0][i]
                }
                
                # Parse speakers back from JSON
                if "speakers" in result["metadata"]:
                    try:
                        result["metadata"]["speakers"] = json.loads(result["metadata"]["speakers"])
                    except:
                        result["metadata"]["speakers"] = []
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def search_by_text(self, query_text: str, embedding_model, n_results: int = 10,
                      episode_filter: Optional[str] = None) -> List[Dict]:
        """
        Search using text query (generates embedding internally).
        
        Args:
            query_text: Text query
            embedding_model: Model to generate query embedding
            n_results: Number of results to return
            episode_filter: Optional episode title to filter by
            
        Returns:
            List of search results with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = embedding_model.encode([query_text])[0]
            
            # Search using embedding
            return self.search(query_embedding, n_results, episode_filter)
            
        except Exception as e:
            print(f"Error in text search: {e}")
            return []
    
    def search_multi_episode(self, query_embedding: np.ndarray, 
                           n_results_per_episode: int = 5) -> Dict[str, List[Dict]]:
        """
        Search across multiple episodes and group results by episode.
        
        Args:
            query_embedding: Query embedding vector
            n_results_per_episode: Number of results per episode
            
        Returns:
            Dictionary mapping episode titles to search results
        """
        try:
            # Get all available episodes
            episodes = self.get_episodes()
            
            results_by_episode = {}
            
            for episode in episodes:
                episode_results = self.search(
                    query_embedding=query_embedding,
                    n_results=n_results_per_episode,
                    episode_filter=episode
                )
                
                if episode_results:
                    results_by_episode[episode] = episode_results
            
            return results_by_episode
            
        except Exception as e:
            print(f"Error in multi-episode search: {e}")
            return {}
    
    def get_episodes(self) -> List[str]:
        """Get list of all episode titles in the database."""
        try:
            # Get all documents with metadata
            results = self.collection.get(include=["metadatas"])
            
            # Extract unique episode titles
            episodes = set()
            for metadata in results["metadatas"]:
                if "episode_title" in metadata:
                    episodes.add(metadata["episode_title"])
            
            return sorted(list(episodes))
            
        except Exception as e:
            print(f"Error getting episodes: {e}")
            return []
    
    def get_episode_stats(self) -> Dict[str, Dict]:
        """Get statistics for each episode in the database."""
        try:
            episodes = self.get_episodes()
            stats = {}
            
            for episode in episodes:
                # Get all chunks for this episode
                results = self.collection.get(
                    where={"episode_title": episode},
                    include=["metadatas"]
                )
                
                if results["metadatas"]:
                    chunk_count = len(results["metadatas"])
                    durations = [meta.get("duration", 0) for meta in results["metadatas"]]
                    total_duration = sum(durations)
                    
                    stats[episode] = {
                        "chunk_count": chunk_count,
                        "total_duration": total_duration,
                        "average_chunk_duration": total_duration / chunk_count if chunk_count > 0 else 0
                    }
            
            return stats
            
        except Exception as e:
            print(f"Error getting episode stats: {e}")
            return {}
    
    def delete_episode(self, episode_title: str):
        """Delete all chunks for a specific episode."""
        try:
            # Get all IDs for the episode
            results = self.collection.get(
                where={"episode_title": episode_title},
                include=["ids"]
            )
            
            if results["ids"]:
                # Delete the chunks
                self.collection.delete(ids=results["ids"])
                print(f"Deleted {len(results['ids'])} chunks for episode: {episode_title}")
            else:
                print(f"No chunks found for episode: {episode_title}")
                
        except Exception as e:
            print(f"Error deleting episode: {e}")
    
    def clear_all(self):
        """Clear all data from the vector store."""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate empty collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Podcast transcript chunks with timestamps"}
            )
            
            print("Cleared all data from vector store.")
            
        except Exception as e:
            print(f"Error clearing vector store: {e}")
    
    def get_collection_size(self) -> int:
        """Get the number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting collection size: {e}")
            return 0
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for display."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def export_data(self, filepath: str):
        """Export all data to a file."""
        try:
            results = self.collection.get(include=["documents", "metadatas", "embeddings"])
            
            export_data = {
                "collection_name": self.collection_name,
                "data": results
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(export_data, f)
            
            print(f"Exported data to {filepath}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def import_data(self, filepath: str):
        """Import data from a file."""
        try:
            with open(filepath, 'rb') as f:
                import_data = pickle.load(f)
            
            data = import_data["data"]
            
            # Add data to collection
            self.collection.add(
                ids=data["ids"],
                documents=data["documents"],
                metadatas=data["metadatas"],
                embeddings=data["embeddings"]
            )
            
            print(f"Imported {len(data['ids'])} chunks from {filepath}")
            
        except Exception as e:
            print(f"Error importing data: {e}")


class RetrievalSystem:
    """High-level retrieval system that combines vector search with additional features."""
    
    def __init__(self, vector_store: PodcastVectorStore, text_processor):
        """
        Initialize the retrieval system.
        
        Args:
            vector_store: Vector store instance
            text_processor: Text processor instance
        """
        self.vector_store = vector_store
        self.text_processor = text_processor
    
    def search_with_context(self, query: str, n_results: int = 10, 
                           context_window: float = 30.0) -> List[Dict]:
        """
        Search with additional context around found chunks.
        
        Args:
            query: Search query
            n_results: Number of results
            context_window: Seconds of context to include around each result
            
        Returns:
            Search results with extended context
        """
        # Perform initial search
        results = self.vector_store.search_by_text(
            query, self.text_processor.embedding_model, n_results
        )
        
        # Add context to each result
        for result in results:
            # Get surrounding chunks for context
            start_time = result["metadata"]["start_time"] - context_window
            end_time = result["metadata"]["end_time"] + context_window
            
            # Search for overlapping chunks
            context_query = np.zeros(384)  # Dummy embedding for metadata search
            context_results = self.vector_store.collection.get(
                where={
                    "episode_title": result["metadata"]["episode_title"],
                    "$and": [
                        {"start_time": {"$gte": start_time}},
                        {"end_time": {"$lte": end_time}}
                    ]
                },
                include=["documents", "metadatas"]
            )
            
            # Combine context
            if context_results["documents"]:
                context_text = " ".join(context_results["documents"])
                result["extended_context"] = context_text
            
        return results
    
    def search_temporal_correlation(self, query: str, time_threshold: float = 300.0) -> List[Dict]:
        """
        Find temporally correlated segments across episodes.
        
        Args:
            query: Search query
            time_threshold: Time threshold for correlation (seconds)
            
        Returns:
            Temporally correlated search results
        """
        # Get initial results
        results = self.vector_store.search_by_text(
            query, self.text_processor.embedding_model, n_results=20
        )
        
        # Group by temporal proximity and topic similarity
        correlated_groups = []
        used_results = set()
        
        for i, result in enumerate(results):
            if i in used_results:
                continue
                
            group = [result]
            used_results.add(i)
            
            for j, other_result in enumerate(results[i+1:], i+1):
                if j in used_results:
                    continue
                
                # Check if they're from same episode and temporally close
                if (result["metadata"]["episode_title"] == other_result["metadata"]["episode_title"] and
                    abs(result["metadata"]["start_time"] - other_result["metadata"]["start_time"]) <= time_threshold):
                    group.append(other_result)
                    used_results.add(j)
            
            if len(group) > 1:  # Only include groups with multiple segments
                correlated_groups.append(group)
        
        return correlated_groups
    
    def get_episode_summary(self, episode_title: str, n_key_segments: int = 5) -> Dict:
        """
        Generate a summary of an episode using key segments.
        
        Args:
            episode_title: Episode title
            n_key_segments: Number of key segments to include
            
        Returns:
            Episode summary with key segments
        """
        # Get all segments for the episode
        all_segments = self.vector_store.collection.get(
            where={"episode_title": episode_title},
            include=["documents", "metadatas"]
        )
        
        if not all_segments["documents"]:
            return {"error": f"No segments found for episode: {episode_title}"}
        
        # Use embedding similarity to find diverse key segments
        segment_texts = all_segments["documents"]
        embeddings = self.text_processor.generate_embeddings(segment_texts)
        
        # Simple diversity selection: pick segments that are most different from each other
        selected_indices = [0]  # Start with first segment
        
        for _ in range(min(n_key_segments - 1, len(embeddings) - 1)):
            max_min_distance = 0
            best_idx = 0
            
            for i, emb in enumerate(embeddings):
                if i in selected_indices:
                    continue
                
                # Find minimum distance to already selected segments
                min_distance = min([
                    np.linalg.norm(emb - embeddings[j]) for j in selected_indices
                ])
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = i
            
            selected_indices.append(best_idx)
        
        # Build summary
        key_segments = []
        for idx in sorted(selected_indices):
            segment = {
                "text": segment_texts[idx],
                "metadata": all_segments["metadatas"][idx]
            }
            key_segments.append(segment)
        
        summary = {
            "episode_title": episode_title,
            "total_segments": len(segment_texts),
            "key_segments": key_segments,
            "total_duration": max([meta.get("end_time", 0) for meta in all_segments["metadatas"]])
        }
        
        return summary