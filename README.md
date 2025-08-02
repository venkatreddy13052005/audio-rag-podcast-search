# 🎙️ Audio-to-Text RAG for Podcast Search

A comprehensive multimodal RAG (Retrieval-Augmented Generation) system that processes audio podcasts, converts them to searchable text, and enables users to query specific topics mentioned across multiple episodes with precise timestamp references and speaker identification.

![Podcast RAG System](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Live Demo

**[Try the Demo on HuggingFace Spaces →](https://huggingface.co/spaces/your-username/podcast-rag-system)**

## ✨ Features

### 🎯 Core Capabilities
- **High-Accuracy Speech-to-Text**: Powered by OpenAI Whisper for precise transcription
- **Speaker Diarization**: Automatic identification and separation of different speakers
- **Semantic Search**: Vector-based search using Sentence Transformers embeddings
- **Multi-Episode Search**: Query across entire podcast libraries
- **Timestamp References**: Precise audio segment location for every result
- **Contextual Retrieval**: Extended context around search results
- **Real-time Analytics**: Performance metrics and database statistics

### 🔧 Technical Features
- **Audio Preprocessing**: Noise reduction and audio normalization
- **Smart Chunking**: Multiple strategies (semantic, time-based, sentence-based)
- **Vector Database**: Persistent storage with ChromaDB
- **Cross-Episode Correlation**: Find related topics across different episodes
- **Performance Evaluation**: Comprehensive benchmarking suite
- **Scalable Architecture**: Modular design for easy extension

## 🏗️ System Architecture

```mermaid
graph TB
    A[Audio Upload] --> B[Audio Preprocessing]
    B --> C[Speech-to-Text (Whisper)]
    B --> D[Speaker Diarization]
    C --> E[Text Chunking]
    D --> E
    E --> F[Embedding Generation]
    F --> G[Vector Database (ChromaDB)]
    H[Search Query] --> I[Query Embedding]
    I --> J[Vector Search]
    G --> J
    J --> K[Retrieval & Ranking]
    K --> L[Results with Timestamps]
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio processing)
- 4GB+ RAM (for AI models)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/podcast-rag-system.git
cd podcast-rag-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg** (if not already installed)
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser** to `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload and Process Episodes

1. Navigate to the **"Upload & Process"** tab
2. Choose an audio file (MP3, WAV, FLAC, M4A, OGG, MP4)
3. Enter a descriptive episode title
4. Click **"Process Episode"** and wait for completion

**Processing Steps:**
- Audio preprocessing and noise reduction
- Speech-to-text transcription with Whisper
- Speaker identification and diarization
- Text chunking with timestamp preservation
- Embedding generation and vector storage

### 2. Search Across Episodes

1. Go to the **"Search"** tab
2. Enter your search query in natural language
3. Choose search type:
   - **Semantic Search**: Standard vector similarity search
   - **Multi-Episode Search**: Search across all episodes simultaneously
   - **Contextual Search**: Include surrounding context for better understanding
4. Adjust parameters as needed
5. View results with timestamps, speakers, and relevance scores

### 3. Analyze Your Data

The **"Analytics"** tab provides:
- Episode statistics and duration breakdowns
- Chunk distribution analysis
- Database overview and metrics
- Visual charts and summaries

### 4. Evaluate Performance

The **"Evaluation"** tab offers:
- Real-time performance metrics
- Comprehensive benchmark suite
- Latency and relevance analysis
- System scalability testing

## 🛠️ Technical Implementation

### Audio Processing Pipeline

```python
from src.audio_processor import AudioProcessor

# Initialize processor
processor = AudioProcessor(whisper_model_size="base")

# Process episode
result = processor.process_podcast_episode(
    audio_file_path="episode.mp3",
    episode_title="Tech Talk #1"
)

# Extract segments with metadata
segments = processor.extract_segments_with_metadata(result)
```

### Text Chunking Strategies

```python
from src.text_processor import TextProcessor

text_processor = TextProcessor()

# Semantic chunking (time-based)
chunks = text_processor.chunk_transcript_semantic(
    segments, 
    chunk_size_seconds=60.0,
    overlap_seconds=10.0
)

# Sentence-based chunking
chunks = text_processor.chunk_transcript_sentence_based(
    segments,
    max_sentences=10,
    overlap_sentences=2
)

# Topic-based chunking (experimental)
chunks = text_processor.chunk_transcript_topic_based(
    segments,
    similarity_threshold=0.7
)
```

### Vector Search

```python
from src.vector_store import PodcastVectorStore

# Initialize vector store
vector_store = PodcastVectorStore()

# Add processed chunks
vector_store.add_chunks(chunks, embeddings)

# Search
results = vector_store.search_by_text(
    "machine learning algorithms",
    embedding_model,
    n_results=10
)
```

## 📊 Performance Metrics

The system includes comprehensive evaluation capabilities:

- **Search Latency**: Average response time < 100ms
- **Relevance Scoring**: Cosine similarity-based ranking
- **Precision@K**: Top-K result accuracy
- **Recall@K**: Coverage of relevant results
- **NDCG**: Normalized Discounted Cumulative Gain
- **Temporal Coverage**: Time span analysis
- **Cross-Episode Correlation**: Topic relationship mapping

## 🔧 Configuration

### Model Configuration

```python
# Whisper model sizes (accuracy vs speed tradeoff)
WHISPER_MODELS = {
    "tiny": "Fastest, lower accuracy",
    "base": "Balanced (default)",
    "small": "Better accuracy",
    "medium": "High accuracy",
    "large": "Best accuracy, slower"
}

# Embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "Fast, 384 dimensions",
    "all-mpnet-base-v2": "Better quality, 768 dimensions",
    "multi-qa-MiniLM-L6-cos-v1": "Optimized for Q&A"
}
```

### Chunking Parameters

```python
CHUNKING_CONFIG = {
    "semantic": {
        "chunk_size_seconds": 60.0,
        "overlap_seconds": 10.0
    },
    "sentence": {
        "max_sentences": 10,
        "overlap_sentences": 2
    },
    "topic": {
        "similarity_threshold": 0.7
    }
}
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```bash
# Build image
docker build -t podcast-rag .

# Run container
docker run -p 8501:8501 podcast-rag
```

### HuggingFace Spaces
1. Fork this repository
2. Connect to HuggingFace Spaces
3. Deploy with automatic builds

### Cloud Deployment (AWS/GCP/Azure)
- Use the provided Dockerfile
- Configure persistent storage for ChromaDB
- Set appropriate memory limits (4GB+ recommended)

## 📁 Project Structure

```
podcast-rag-system/
├── src/
│   ├── __init__.py
│   ├── audio_processor.py      # Audio processing and transcription
│   ├── text_processor.py       # Text chunking and embeddings
│   ├── vector_store.py         # ChromaDB vector storage
│   └── evaluation.py           # Performance metrics
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Dockerfile                  # Container deployment
└── chroma_db/                  # Persistent vector database
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/your-username/podcast-rag-system.git
cd podcast-rag-system
pip install -r requirements.txt
pre-commit install  # For code formatting
```

### Running Tests
```bash
pytest tests/
python -m pytest --cov=src tests/
```

## 📈 Roadmap

- [ ] **Multi-language Support**: Extend beyond English transcription
- [ ] **Real-time Processing**: Live podcast ingestion
- [ ] **Advanced Analytics**: Topic modeling and trend analysis
- [ ] **API Integration**: RESTful API for external access
- [ ] **Mobile App**: React Native companion app
- [ ] **Collaborative Features**: Shared podcast libraries
- [ ] **Advanced Speaker ID**: Neural speaker recognition
- [ ] **Summarization**: Automatic episode summaries

## 🔬 Research Applications

This system can be used for:
- **Academic Research**: Analyze interview transcripts and oral histories
- **Market Research**: Process focus groups and customer interviews
- **Journalism**: Search through interview archives
- **Education**: Create searchable lecture libraries
- **Legal**: Analyze depositions and court proceedings
- **Healthcare**: Process patient consultation recordings

## 📚 Technical Papers and References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) - Whisper paper
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) - Sentence Transformers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - RAG methodology

## 🛡️ Privacy and Security

- **Local Processing**: All audio processing happens locally
- **No Data Sharing**: Audio files are not sent to external services
- **Secure Storage**: Vector embeddings are stored locally with ChromaDB
- **Privacy Controls**: Easy data deletion and export capabilities

## ⚠️ Limitations

- **Model Dependencies**: Requires downloading large AI models (1-3GB)
- **Processing Time**: Initial episode processing can take 2-5 minutes
- **Language Support**: Currently optimized for English audio
- **Hardware Requirements**: Minimum 4GB RAM recommended
- **Audio Quality**: Performance depends on input audio clarity

## 🎓 Educational Use

This project serves as an excellent learning resource for:
- **RAG System Development**: Complete end-to-end implementation
- **Multimodal AI**: Audio and text processing integration
- **Vector Databases**: Practical ChromaDB usage
- **Streamlit Development**: Modern web app creation
- **Evaluation Metrics**: Comprehensive performance measurement

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the Whisper speech recognition model
- **Sentence Transformers** team for embedding models
- **ChromaDB** for vector database capabilities
- **Streamlit** for the amazing web framework
- **PyAnnote** for speaker diarization tools
- **HuggingFace** for model hosting and deployment

## 📞 Support

- **Documentation**: [Full Documentation](https://your-docs-site.com)
- **Issues**: [GitHub Issues](https://github.com/your-username/podcast-rag-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/podcast-rag-system/discussions)
- **Email**: support@your-domain.com

---

**Built with ❤️ for the podcast community and RAG enthusiasts**

*Star ⭐ this repository if you find it useful!*