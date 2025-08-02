"""
Demo script to test the Audio-to-Text RAG system with sample functionality.
This script demonstrates the core capabilities without requiring audio files.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.audio_processor import AudioProcessor
from src.text_processor import TextProcessor
from src.vector_store import PodcastVectorStore, RetrievalSystem
from src.evaluation import EvaluationMetrics, BenchmarkSuite


def create_sample_data():
    """Create sample podcast data for demonstration."""
    
    sample_episodes = [
        {
            "title": "Tech Talk: Machine Learning Fundamentals",
            "segments": [
                {
                    "text": "Welcome to Tech Talk! Today we're diving deep into machine learning fundamentals. I'm your host Sarah, and I'm joined by Dr. Alex Chen, a machine learning researcher at Stanford.",
                    "start_time": 0.0,
                    "end_time": 8.5,
                    "speaker": "SPEAKER_01"
                },
                {
                    "text": "Thanks for having me, Sarah. Machine learning is essentially about teaching computers to learn patterns from data without being explicitly programmed for every scenario.",
                    "start_time": 8.5,
                    "end_time": 18.2,
                    "speaker": "SPEAKER_02"
                },
                {
                    "text": "That's fascinating. Can you explain the difference between supervised and unsupervised learning?",
                    "start_time": 18.2,
                    "end_time": 24.1,
                    "speaker": "SPEAKER_01"
                },
                {
                    "text": "Absolutely. Supervised learning uses labeled data to train models. Think of it like learning with a teacher who provides the correct answers. Unsupervised learning finds hidden patterns in data without labels.",
                    "start_time": 24.1,
                    "end_time": 36.8,
                    "speaker": "SPEAKER_02"
                },
                {
                    "text": "Neural networks are a cornerstone of deep learning. They're inspired by how the human brain processes information through interconnected neurons.",
                    "start_time": 36.8,
                    "end_time": 46.3,
                    "speaker": "SPEAKER_02"
                }
            ]
        },
        {
            "title": "Climate Science Weekly: Ocean Acidification",
            "segments": [
                {
                    "text": "This is Climate Science Weekly. I'm Dr. Maria Rodriguez, and today we're discussing ocean acidification, often called the other CO2 problem.",
                    "start_time": 0.0,
                    "end_time": 9.2,
                    "speaker": "SPEAKER_01"
                },
                {
                    "text": "Ocean acidification occurs when the ocean absorbs carbon dioxide from the atmosphere. This process lowers the pH of seawater, making it more acidic.",
                    "start_time": 9.2,
                    "end_time": 19.7,
                    "speaker": "SPEAKER_01"
                },
                {
                    "text": "The impacts on marine ecosystems are profound. Coral reefs, shellfish, and many other marine organisms struggle to build and maintain their calcium carbonate structures in more acidic waters.",
                    "start_time": 19.7,
                    "end_time": 32.4,
                    "speaker": "SPEAKER_01"
                },
                {
                    "text": "We're seeing significant changes in coral reef ecosystems worldwide. The Great Barrier Reef has experienced multiple bleaching events linked to both warming temperatures and changing ocean chemistry.",
                    "start_time": 32.4,
                    "end_time": 44.8,
                    "speaker": "SPEAKER_01"
                }
            ]
        },
        {
            "title": "Startup Stories: Building a SaaS Company",
            "segments": [
                {
                    "text": "Welcome to Startup Stories! I'm Jake Williams, and today I'm talking with Lisa Park, founder of DataFlow Analytics, about building a successful SaaS company.",
                    "start_time": 0.0,
                    "end_time": 10.3,
                    "speaker": "SPEAKER_01"
                },
                {
                    "text": "Thanks Jake. When I started DataFlow Analytics, I had no idea how challenging it would be to find product-market fit. We pivoted three times before finding our niche in financial analytics.",
                    "start_time": 10.3,
                    "end_time": 22.7,
                    "speaker": "SPEAKER_02"
                },
                {
                    "text": "Funding was crucial for our growth. We raised a seed round of $2 million from angel investors who understood the fintech space. The key was demonstrating clear customer traction.",
                    "start_time": 22.7,
                    "end_time": 35.1,
                    "speaker": "SPEAKER_02"
                },
                {
                    "text": "Building a strong engineering team was essential. We focused on hiring senior developers who could scale our infrastructure as we grew from 10 to 10,000 customers.",
                    "start_time": 35.1,
                    "end_time": 46.8,
                    "speaker": "SPEAKER_02"
                }
            ]
        }
    ]
    
    return sample_episodes


def format_segments_for_processor(episode_data):
    """Format episode data for the text processor."""
    formatted_segments = []
    
    for segment in episode_data["segments"]:
        formatted_segment = {
            "text": segment["text"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "speaker": segment["speaker"],
            "episode_title": episode_data["title"],
            "audio_file": f"demo_{episode_data['title'].lower().replace(' ', '_')}.wav"
        }
        formatted_segments.append(formatted_segment)
    
    return formatted_segments


def main():
    """Main demo function."""
    
    print("🎙️ Audio-to-Text RAG System Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    try:
        text_processor = TextProcessor(embedding_model_name="all-MiniLM-L6-v2")
        vector_store = PodcastVectorStore(
            persist_directory="./demo_chroma_db",
            collection_name="demo_podcast_transcripts"
        )
        retrieval_system = RetrievalSystem(vector_store, text_processor)
        evaluation_metrics = EvaluationMetrics()
        
        print("✅ Components initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        return
    
    # Create and process sample data
    print("\n2. Processing sample podcast episodes...")
    sample_episodes = create_sample_data()
    
    all_chunks = []
    for episode_data in sample_episodes:
        print(f"   Processing: {episode_data['title']}")
        
        # Format segments
        segments = format_segments_for_processor(episode_data)
        
        # Create chunks
        chunks = text_processor.chunk_transcript_semantic(
            segments, 
            chunk_size_seconds=30.0,
            overlap_seconds=5.0
        )
        
        all_chunks.extend(chunks)
        print(f"   Created {len(chunks)} chunks")
    
    # Generate embeddings and store chunks
    print(f"\n3. Generating embeddings for {len(all_chunks)} chunks...")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    embeddings = text_processor.generate_embeddings(chunk_texts)
    
    print("4. Storing chunks in vector database...")
    vector_store.add_chunks(all_chunks, embeddings)
    
    print("✅ Data processing complete!")
    
    # Demonstrate search capabilities
    print("\n5. Testing search capabilities...")
    test_queries = [
        "machine learning algorithms and neural networks",
        "climate change and ocean acidification",
        "startup funding and investors",
        "coral reefs and marine ecosystems",
        "SaaS company product market fit"
    ]
    
    search_results_summary = []
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        # Measure search time
        start_time = time.time()
        results = vector_store.search_by_text(
            query, text_processor.embedding_model, n_results=3
        )
        search_time = time.time() - start_time
        
        print(f"   Found {len(results)} results in {search_time:.3f}s")
        
        # Calculate relevance scores
        relevance_scores = evaluation_metrics.calculate_relevance_score(
            query, results, text_processor.embedding_model
        )
        
        if results:
            best_result = results[0]
            print(f"   Best match: {best_result['metadata']['episode_title']}")
            print(f"   Similarity: {best_result['similarity_score']:.2%}")
            print(f"   Timestamp: {best_result['metadata']['timestamp_formatted']}")
            print(f"   Text preview: {best_result['text'][:100]}...")
            
            search_results_summary.append({
                "query": query,
                "num_results": len(results),
                "search_time": search_time,
                "best_similarity": best_result['similarity_score'],
                "best_episode": best_result['metadata']['episode_title']
            })
    
    # Multi-episode search demonstration
    print("\n6. Testing multi-episode search...")
    query_embedding = text_processor.embedding_model.encode(["artificial intelligence and data science"])[0]
    multi_results = vector_store.search_multi_episode(query_embedding, n_results_per_episode=2)
    
    print(f"   Multi-episode search found results in {len(multi_results)} episodes:")
    for episode, results in multi_results.items():
        print(f"   - {episode}: {len(results)} results")
    
    # Performance evaluation
    print("\n7. Performance evaluation...")
    performance_summary = evaluation_metrics.get_performance_summary()
    
    print(f"   Average search latency: {performance_summary['latency_stats']['mean']:.3f}s")
    print(f"   Average relevance score: {performance_summary['relevance_stats']['mean']:.2%}")
    print(f"   Total searches performed: {performance_summary['latency_stats']['count']}")
    
    # Database statistics
    print("\n8. Database statistics...")
    episode_stats = vector_store.get_episode_stats()
    collection_size = vector_store.get_collection_size()
    episodes = vector_store.get_episodes()
    
    print(f"   Total chunks in database: {collection_size}")
    print(f"   Number of episodes: {len(episodes)}")
    
    for episode, stats in episode_stats.items():
        print(f"   - {episode}: {stats['chunk_count']} chunks, {stats['total_duration']:.1f}s duration")
    
    # Summary report
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY REPORT")
    print("=" * 50)
    
    print(f"✅ Processed {len(sample_episodes)} podcast episodes")
    print(f"✅ Created {len(all_chunks)} searchable text chunks")
    print(f"✅ Generated {len(embeddings)} embedding vectors")
    print(f"✅ Tested {len(test_queries)} search queries")
    print(f"✅ Average search time: {performance_summary['latency_stats']['mean']:.3f}s")
    print(f"✅ Average relevance score: {performance_summary['relevance_stats']['mean']:.2%}")
    
    print("\n🎯 Search Results Summary:")
    for result in search_results_summary:
        print(f"   '{result['query'][:30]}...' → {result['num_results']} results, "
              f"{result['search_time']:.3f}s, {result['best_similarity']:.2%} similarity")
    
    print("\n🚀 Demo completed successfully!")
    print("💡 The system is ready for real podcast audio files!")
    
    # Cleanup option
    cleanup = input("\n🗑️ Would you like to clean up the demo database? (y/n): ").lower().strip()
    if cleanup == 'y':
        vector_store.clear_all()
        print("✅ Demo database cleaned up.")
        
        # Remove demo database directory
        import shutil
        if os.path.exists("./demo_chroma_db"):
            shutil.rmtree("./demo_chroma_db")
            print("✅ Demo database directory removed.")


if __name__ == "__main__":
    main()