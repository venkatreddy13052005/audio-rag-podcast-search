"""
Streamlit application for the Audio-to-Text RAG system for podcast search.
"""

import streamlit as st
import os
import tempfile
import time
from typing import List, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our modules
from src.audio_processor import AudioProcessor, format_timestamp
from src.text_processor import TextProcessor
from src.vector_store import PodcastVectorStore, RetrievalSystem
from src.evaluation import EvaluationMetrics, BenchmarkSuite


# Page configuration
st.set_page_config(
    page_title="Podcast Search RAG System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}

.search-result {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e1e4e8;
    margin: 0.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.timestamp-badge {
    background-color: #667eea;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
    font-weight: bold;
}

.speaker-badge {
    background-color: #764ba2;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
    margin-left: 0.5rem;
}

.episode-title {
    color: #667eea;
    font-weight: bold;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_episodes' not in st.session_state:
    st.session_state.processed_episodes = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retrieval_system' not in st.session_state:
    st.session_state.retrieval_system = None
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = None
if 'text_processor' not in st.session_state:
    st.session_state.text_processor = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = EvaluationMetrics()


@st.cache_resource
def initialize_models():
    """Initialize and cache the models."""
    with st.spinner("Loading models... This may take a few minutes on first run."):
        try:
            # Initialize processors
            audio_processor = AudioProcessor(whisper_model_size="base")
            text_processor = TextProcessor(embedding_model_name="all-MiniLM-L6-v2")
            
            # Initialize vector store
            vector_store = PodcastVectorStore(
                persist_directory="./chroma_db",
                collection_name="podcast_transcripts"
            )
            
            # Initialize retrieval system
            retrieval_system = RetrievalSystem(vector_store, text_processor)
            
            return audio_processor, text_processor, vector_store, retrieval_system
            
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            return None, None, None, None


def process_audio_file(audio_file, episode_title: str):
    """Process uploaded audio file."""
    if st.session_state.audio_processor is None:
        st.error("Audio processor not initialized!")
        return None
    
    with st.spinner(f"Processing audio file: {episode_title}"):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_file_path = tmp_file.name
            
            # Process the episode
            progress_bar = st.progress(0)
            progress_bar.progress(0.2)
            
            st.info("Step 1/4: Transcribing audio...")
            processing_result = st.session_state.audio_processor.process_podcast_episode(
                tmp_file_path, episode_title
            )
            progress_bar.progress(0.4)
            
            st.info("Step 2/4: Extracting segments...")
            segments = st.session_state.audio_processor.extract_segments_with_metadata(processing_result)
            progress_bar.progress(0.6)
            
            st.info("Step 3/4: Creating text chunks...")
            chunks = st.session_state.text_processor.chunk_transcript_semantic(
                segments, chunk_size_seconds=60.0, overlap_seconds=10.0
            )
            progress_bar.progress(0.8)
            
            st.info("Step 4/4: Generating embeddings and storing...")
            if chunks:
                chunk_texts = [chunk["text"] for chunk in chunks]
                embeddings = st.session_state.text_processor.generate_embeddings(chunk_texts)
                st.session_state.vector_store.add_chunks(chunks, embeddings)
            
            progress_bar.progress(1.0)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Update session state
            episode_info = {
                "title": episode_title,
                "duration": processing_result.get("duration", 0),
                "chunks": len(chunks),
                "segments": len(segments)
            }
            st.session_state.processed_episodes.append(episode_info)
            
            return episode_info
            
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return None


def display_search_results(results: List[Dict], query: str):
    """Display search results in a formatted way."""
    if not results:
        st.warning("No results found for your query.")
        return
    
    st.success(f"Found {len(results)} relevant segments")
    
    for i, result in enumerate(results):
        with st.container():
            st.markdown('<div class="search-result">', unsafe_allow_html=True)
            
            # Header with episode title and timestamp
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f'<div class="episode-title">{result["metadata"]["episode_title"]}</div>', 
                          unsafe_allow_html=True)
            
            with col2:
                timestamp = result["metadata"]["timestamp_formatted"]
                st.markdown(f'<span class="timestamp-badge">⏰ {timestamp}</span>', 
                          unsafe_allow_html=True)
            
            with col3:
                similarity = result["similarity_score"]
                st.markdown(f'<span class="timestamp-badge">📊 {similarity:.2%}</span>', 
                          unsafe_allow_html=True)
            
            # Content
            st.markdown(f"**Text:** {result['text']}")
            
            # Speakers
            speakers = result["metadata"].get("speakers", [])
            if speakers:
                speaker_text = ", ".join(speakers)
                st.markdown(f'<span class="speaker-badge">👥 {speaker_text}</span>', 
                          unsafe_allow_html=True)
            
            # Duration
            duration = result["metadata"].get("duration", 0)
            st.caption(f"Duration: {duration:.1f}s | Start: {result['metadata']['start_time']:.1f}s | "
                      f"End: {result['metadata']['end_time']:.1f}s")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎙️ Podcast Search RAG System</h1>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    **Audio-to-Text RAG for Podcast Search** - Upload podcast episodes, search across transcripts, 
    and find specific topics with timestamp references and speaker identification.
    """)
    
    # Initialize models
    if st.session_state.audio_processor is None:
        audio_processor, text_processor, vector_store, retrieval_system = initialize_models()
        if audio_processor:
            st.session_state.audio_processor = audio_processor
            st.session_state.text_processor = text_processor
            st.session_state.vector_store = vector_store
            st.session_state.retrieval_system = retrieval_system
            st.success("Models initialized successfully!")
        else:
            st.error("Failed to initialize models. Please refresh the page.")
            return
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Display current database stats
        if st.session_state.vector_store:
            collection_size = st.session_state.vector_store.get_collection_size()
            episodes = st.session_state.vector_store.get_episodes()
            
            st.metric("Total Chunks", collection_size)
            st.metric("Episodes Processed", len(episodes))
            
            if episodes:
                st.subheader("Episodes in Database")
                for episode in episodes:
                    st.write(f"• {episode}")
        
        # Clear database
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear_all()
                st.session_state.processed_episodes = []
                st.success("Database cleared!")
                st.experimental_rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "🔍 Search", "📊 Analytics", "⚡ Evaluation"])
    
    with tab1:
        st.header("Upload Podcast Episodes")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['mp3', 'wav', 'flac', 'm4a', 'ogg', 'mp4'],
                help="Supported formats: MP3, WAV, FLAC, M4A, OGG, MP4"
            )
            
            episode_title = st.text_input(
                "Episode Title",
                placeholder="Enter a descriptive title for this episode"
            )
        
        with col2:
            st.info("""
            **Processing Steps:**
            1. Audio preprocessing & noise reduction
            2. Speech-to-text with Whisper
            3. Speaker diarization
            4. Text chunking with timestamps
            5. Embedding generation & storage
            """)
        
        if uploaded_file and episode_title:
            if st.button("🚀 Process Episode", type="primary"):
                episode_info = process_audio_file(uploaded_file, episode_title)
                
                if episode_info:
                    st.success("Episode processed successfully! 🎉")
                    
                    # Display processing results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Duration", f"{episode_info['duration']:.1f}s")
                    with col2:
                        st.metric("Segments", episode_info['segments'])
                    with col3:
                        st.metric("Chunks", episode_info['chunks'])
                    with col4:
                        avg_chunk_size = episode_info['duration'] / episode_info['chunks'] if episode_info['chunks'] > 0 else 0
                        st.metric("Avg Chunk Size", f"{avg_chunk_size:.1f}s")
    
    with tab2:
        st.header("Search Podcast Transcripts")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="Enter your search query (e.g., 'machine learning algorithms', 'climate change policy')",
                help="Use natural language to search across all processed episodes"
            )
        
        with col2:
            search_type = st.selectbox(
                "Search Type",
                ["Semantic Search", "Multi-Episode Search", "Contextual Search"],
                help="Choose the type of search to perform"
            )
        
        # Search parameters
        with st.expander("🔧 Search Parameters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_results = st.slider("Number of Results", 5, 50, 10)
            
            with col2:
                episode_filter = st.selectbox(
                    "Filter by Episode",
                    ["All Episodes"] + st.session_state.vector_store.get_episodes() if st.session_state.vector_store else ["All Episodes"]
                )
            
            with col3:
                if search_type == "Contextual Search":
                    context_window = st.slider("Context Window (seconds)", 10, 120, 30)
        
        # Perform search
        if query:
            search_start_time = time.time()
            
            try:
                episode_filter_val = None if episode_filter == "All Episodes" else episode_filter
                
                if search_type == "Semantic Search":
                    results = st.session_state.vector_store.search_by_text(
                        query, st.session_state.text_processor.embedding_model, 
                        n_results, episode_filter_val
                    )
                elif search_type == "Multi-Episode Search":
                    query_embedding = st.session_state.text_processor.embedding_model.encode([query])[0]
                    multi_results = st.session_state.vector_store.search_multi_episode(
                        query_embedding, n_results_per_episode=5
                    )
                    # Flatten results
                    results = []
                    for episode, episode_results in multi_results.items():
                        results.extend(episode_results)
                    # Sort by similarity score
                    results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:n_results]
                
                elif search_type == "Contextual Search":
                    results = st.session_state.retrieval_system.search_with_context(
                        query, n_results, context_window
                    )
                
                search_time = time.time() - search_start_time
                
                # Record metrics
                st.session_state.evaluation_metrics.search_times.append(search_time)
                if results:
                    relevance_scores = st.session_state.evaluation_metrics.calculate_relevance_score(
                        query, results, st.session_state.text_processor.embedding_model
                    )
                
                # Display search info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Search Time", f"{search_time:.3f}s")
                with col2:
                    st.metric("Results Found", len(results))
                with col3:
                    if results:
                        avg_relevance = sum(relevance_scores) / len(relevance_scores)
                        st.metric("Avg Relevance", f"{avg_relevance:.2%}")
                
                # Display results
                display_search_results(results, query)
                
            except Exception as e:
                st.error(f"Search error: {e}")
    
    with tab3:
        st.header("Database Analytics")
        
        if st.session_state.vector_store and st.session_state.vector_store.get_collection_size() > 0:
            # Get episode statistics
            episode_stats = st.session_state.vector_store.get_episode_stats()
            
            if episode_stats:
                # Episode overview
                st.subheader("Episode Overview")
                
                df = pd.DataFrame.from_dict(episode_stats, orient='index').reset_index()
                df.rename(columns={'index': 'episode'}, inplace=True)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_chunks = px.bar(
                        df, x='episode', y='chunk_count',
                        title='Chunks per Episode',
                        labels={'chunk_count': 'Number of Chunks', 'episode': 'Episode'}
                    )
                    fig_chunks.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_chunks, use_container_width=True)
                
                with col2:
                    fig_duration = px.bar(
                        df, x='episode', y='total_duration',
                        title='Duration per Episode (seconds)',
                        labels={'total_duration': 'Duration (seconds)', 'episode': 'Episode'}
                    )
                    fig_duration.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_duration, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Episodes", len(episode_stats))
                with col2:
                    total_chunks = sum([stats['chunk_count'] for stats in episode_stats.values()])
                    st.metric("Total Chunks", total_chunks)
                with col3:
                    total_duration = sum([stats['total_duration'] for stats in episode_stats.values()])
                    st.metric("Total Duration", f"{total_duration/60:.1f} min")
                with col4:
                    avg_chunk_duration = sum([stats['average_chunk_duration'] for stats in episode_stats.values()]) / len(episode_stats)
                    st.metric("Avg Chunk Duration", f"{avg_chunk_duration:.1f}s")
        else:
            st.info("No data available. Please upload and process some episodes first.")
    
    with tab4:
        st.header("System Evaluation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Performance Metrics")
            
            # Get current performance summary
            if st.session_state.evaluation_metrics:
                summary = st.session_state.evaluation_metrics.get_performance_summary()
                
                # Display metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    latency_stats = summary['latency_stats']
                    st.metric(
                        "Avg Search Latency",
                        f"{latency_stats['mean']:.3f}s",
                        f"±{latency_stats['std']:.3f}s"
                    )
                
                with col_b:
                    relevance_stats = summary['relevance_stats']
                    st.metric(
                        "Avg Relevance Score",
                        f"{relevance_stats['mean']:.2%}",
                        f"±{relevance_stats['std']:.2%}"
                    )
                
                with col_c:
                    st.metric(
                        "Total Searches",
                        latency_stats['count']
                    )
        
        with col2:
            st.subheader("Run Benchmark")
            
            if st.button("🚀 Run Comprehensive Benchmark", type="primary"):
                if st.session_state.retrieval_system and st.session_state.vector_store.get_collection_size() > 0:
                    with st.spinner("Running benchmark suite..."):
                        benchmark_suite = BenchmarkSuite(
                            st.session_state.retrieval_system,
                            st.session_state.text_processor
                        )
                        
                        results = benchmark_suite.run_comprehensive_benchmark()
                        
                        # Display results
                        report = benchmark_suite.generate_benchmark_report(results)
                        st.text_area("Benchmark Report", report, height=400)
                        
                        # Save results
                        benchmark_suite.save_benchmark_results(results, "benchmark_results.json")
                        st.success("Benchmark completed! Results saved to benchmark_results.json")
                else:
                    st.warning("Please process some episodes first before running benchmarks.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🎙️ <strong>Podcast Search RAG System</strong> | Built with Streamlit, Whisper, ChromaDB, and Sentence Transformers</p>
        <p>Upload audio → Transcribe → Search → Discover insights across your podcast library</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()