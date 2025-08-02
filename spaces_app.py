"""
Simplified Streamlit application for HuggingFace Spaces deployment.
This version uses lighter models and includes sample data for demonstration.
"""

import streamlit as st
import os
import tempfile
import time
from typing import List, Dict
import pandas as pd

# Import our modules
from src.text_processor import TextProcessor
from src.vector_store import PodcastVectorStore, RetrievalSystem
from src.evaluation import EvaluationMetrics

# Page configuration
st.set_page_config(
    page_title="Podcast Search RAG Demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.search-result {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}

.timestamp-badge {
    background-color: #667eea;
    color: white;
    padding: 0.2rem 0.5rem;
    border-radius: 0.3rem;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'text_processor' not in st.session_state:
    st.session_state.text_processor = None
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False


def get_sample_data():
    """Sample podcast data for demonstration."""
    return [
        {
            "text": "Welcome to Tech Talk! Today we're diving deep into machine learning fundamentals. Machine learning is essentially about teaching computers to learn patterns from data without being explicitly programmed.",
            "start_time": 0.0,
            "end_time": 15.0,
            "episode_title": "Tech Talk: ML Fundamentals",
            "speakers": ["Sarah", "Dr. Alex Chen"],
            "chunk_id": "tech_talk_0_15"
        },
        {
            "text": "Neural networks are a cornerstone of deep learning. They're inspired by how the human brain processes information through interconnected neurons. Deep learning has revolutionized fields like computer vision and natural language processing.",
            "start_time": 15.0,
            "end_time": 35.0,
            "episode_title": "Tech Talk: ML Fundamentals",
            "speakers": ["Dr. Alex Chen"],
            "chunk_id": "tech_talk_15_35"
        },
        {
            "text": "This is Climate Science Weekly. Ocean acidification occurs when the ocean absorbs carbon dioxide from the atmosphere. This process lowers the pH of seawater, making it more acidic and threatening marine ecosystems.",
            "start_time": 0.0,
            "end_time": 18.0,
            "episode_title": "Climate Science Weekly: Ocean Acidification",
            "speakers": ["Dr. Maria Rodriguez"],
            "chunk_id": "climate_0_18"
        },
        {
            "text": "The impacts on marine ecosystems are profound. Coral reefs, shellfish, and many other marine organisms struggle to build and maintain their calcium carbonate structures in more acidic waters. We're seeing significant changes worldwide.",
            "start_time": 18.0,
            "end_time": 40.0,
            "episode_title": "Climate Science Weekly: Ocean Acidification",
            "speakers": ["Dr. Maria Rodriguez"],
            "chunk_id": "climate_18_40"
        },
        {
            "text": "Welcome to Startup Stories! Building a successful SaaS company requires finding product-market fit. We pivoted three times before finding our niche in financial analytics. The key was demonstrating clear customer traction to investors.",
            "start_time": 0.0,
            "end_time": 20.0,
            "episode_title": "Startup Stories: Building a SaaS Company",
            "speakers": ["Jake Williams", "Lisa Park"],
            "chunk_id": "startup_0_20"
        },
        {
            "text": "Funding was crucial for our growth. We raised a seed round of $2 million from angel investors who understood the fintech space. Building a strong engineering team was essential for scaling our infrastructure.",
            "start_time": 20.0,
            "end_time": 40.0,
            "episode_title": "Startup Stories: Building a SaaS Company",
            "speakers": ["Lisa Park"],
            "chunk_id": "startup_20_40"
        }
    ]


@st.cache_resource
def initialize_system():
    """Initialize the RAG system with lightweight models."""
    with st.spinner("Loading models... This may take a moment."):
        try:
            # Use smaller, faster model for demo
            text_processor = TextProcessor(embedding_model_name="all-MiniLM-L6-v2")
            
            vector_store = PodcastVectorStore(
                persist_directory="./demo_chroma_db",
                collection_name="demo_transcripts"
            )
            
            retrieval_system = RetrievalSystem(vector_store, text_processor)
            
            return text_processor, vector_store, retrieval_system
            
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            return None, None, None


def load_sample_data():
    """Load sample data into the vector store."""
    if st.session_state.sample_data_loaded:
        return
    
    with st.spinner("Loading sample podcast data..."):
        sample_chunks = get_sample_data()
        
        # Generate embeddings
        chunk_texts = [chunk["text"] for chunk in sample_chunks]
        embeddings = st.session_state.text_processor.generate_embeddings(chunk_texts)
        
        # Add to vector store
        st.session_state.vector_store.add_chunks(sample_chunks, embeddings)
        st.session_state.sample_data_loaded = True


def format_timestamp(seconds):
    """Format timestamp for display."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def display_search_results(results: List[Dict], query: str):
    """Display search results."""
    if not results:
        st.warning("No results found for your query.")
        return
    
    st.success(f"Found {len(results)} relevant segments")
    
    for i, result in enumerate(results):
        with st.container():
            st.markdown('<div class="search-result">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{result['metadata']['episode_title']}**")
            
            with col2:
                timestamp = format_timestamp(result['metadata']['start_time'])
                st.markdown(f'<span class="timestamp-badge">⏰ {timestamp}</span>', 
                          unsafe_allow_html=True)
            
            with col3:
                similarity = result['similarity_score']
                st.markdown(f'<span class="timestamp-badge">📊 {similarity:.1%}</span>', 
                          unsafe_allow_html=True)
            
            st.markdown(f"_{result['text']}_")
            
            speakers = result['metadata'].get('speakers', [])
            if speakers:
                speaker_text = ", ".join(speakers) if isinstance(speakers, list) else speakers
                st.caption(f"👥 Speakers: {speaker_text}")
            
            st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎙️ Podcast Search RAG Demo</h1>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    **Audio-to-Text RAG for Podcast Search** - This demo showcases semantic search across podcast transcripts 
    with timestamp references and speaker identification. Try searching for topics like "machine learning", 
    "climate change", or "startup funding".
    """)
    
    # Initialize system
    if st.session_state.text_processor is None:
        text_processor, vector_store, retrieval_system = initialize_system()
        if text_processor:
            st.session_state.text_processor = text_processor
            st.session_state.vector_store = vector_store
            st.session_state.retrieval_system = retrieval_system
            st.success("✅ System initialized!")
        else:
            st.error("❌ Failed to initialize system")
            return
    
    # Load sample data
    if not st.session_state.sample_data_loaded:
        load_sample_data()
        st.success("✅ Sample podcast data loaded!")
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 Demo Information")
        
        if st.session_state.vector_store:
            collection_size = st.session_state.vector_store.get_collection_size()
            episodes = st.session_state.vector_store.get_episodes()
            
            st.metric("Sample Chunks", collection_size)
            st.metric("Sample Episodes", len(episodes))
            
            st.subheader("Available Episodes")
            for episode in episodes:
                st.write(f"• {episode}")
        
        st.markdown("---")
        st.markdown("""
        **Sample Topics to Search:**
        - Machine learning algorithms
        - Neural networks and AI
        - Ocean acidification
        - Climate change impacts
        - Startup funding strategies
        - SaaS company building
        """)
    
    # Main search interface
    st.header("🔍 Search Podcast Transcripts")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'machine learning algorithms', 'climate change', 'startup funding'",
            help="Use natural language to search across all podcast episodes"
        )
    
    with col2:
        n_results = st.selectbox("Results to show:", [3, 5, 10], index=0)
    
    # Perform search
    if query:
        start_time = time.time()
        
        try:
            results = st.session_state.vector_store.search_by_text(
                query, st.session_state.text_processor.embedding_model, n_results
            )
            
            search_time = time.time() - start_time
            
            # Display search metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Search Time", f"{search_time:.3f}s")
            with col2:
                st.metric("Results Found", len(results))
            with col3:
                if results:
                    avg_similarity = sum([r['similarity_score'] for r in results]) / len(results)
                    st.metric("Avg Similarity", f"{avg_similarity:.1%}")
            
            # Display results
            display_search_results(results, query)
            
        except Exception as e:
            st.error(f"Search error: {e}")
    
    # Example queries
    st.header("💡 Try These Example Queries")
    
    example_queries = [
        "machine learning and neural networks",
        "ocean acidification and coral reefs", 
        "startup funding and investors",
        "deep learning algorithms",
        "climate change impacts"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(f"'{example}'", key=f"example_{i}"):
                st.experimental_set_query_params(query=example)
                st.experimental_rerun()
    
    # System information
    st.markdown("---")
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        This demonstration showcases a **Retrieval-Augmented Generation (RAG) system** specifically designed for podcast search:
        
        **🔧 Technical Features:**
        - **Speech-to-Text**: OpenAI Whisper for accurate transcription
        - **Semantic Search**: Sentence Transformers for meaning-based search
        - **Vector Database**: ChromaDB for efficient similarity search  
        - **Speaker Identification**: Automatic speaker diarization
        - **Timestamp Mapping**: Precise audio segment location
        
        **📊 Sample Data:**
        - 3 sample podcast episodes covering tech, climate science, and startups
        - 6 searchable text chunks with timestamps and speaker information
        - Semantic embeddings for natural language search
        
        **🚀 Full System Features:**
        - Upload and process real audio files
        - Multiple chunking strategies
        - Cross-episode correlation analysis
        - Performance benchmarking
        - Real-time analytics
        
        Built with Python, Streamlit, ChromaDB, and Hugging Face Transformers.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎙️ <strong>Podcast Search RAG System Demo</strong> | 
        <a href="https://github.com/your-username/podcast-rag-system" target="_blank">GitHub</a> | 
        Built with ❤️ for the podcast community</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()