# AMP-Chat: Antimicrobial Peptide Knowledge Q&A System

A Retrieval-Augmented Generation (RAG) system specialized for antimicrobial peptide (AMP) research, providing intelligent question-answering capabilities based on scientific literature.

## 🔬 Overview

AMP-Chat is an AI-powered knowledge system that helps researchers quickly find information about antimicrobial peptides from scientific papers. The system uses advanced vector retrieval and natural language generation to provide accurate, cited responses about AMP sequences, MIC values, mechanisms of action, and more.

### Key Features

- **Intelligent Document Processing**: Automatically extracts and chunks PDF research papers
- **Vector-Based Retrieval**: Uses FAISS IndexFlatIP for precise similarity search
- **Microorganism-Aware Search**: Recognizes bacterial species and expands queries accordingly
- **Streaming Responses**: Real-time answer generation with source citations
- **Web Interface**: Clean, responsive UI with Markdown rendering
- **Mathematical Modeling**: Generates technical reports with formal mathematical foundations

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│  Text Chunking   │───▶│   Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Retrieval      │◀───│  Vector Index   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │              ┌──────────────────┐
         └─────────────▶│   Generation     │
                        └──────────────────┘
                                 │
                        ┌──────────────────┐
                        │   Response       │
                        └──────────────────┘
```

### Core Components

- **Vector Database**: FAISS IndexFlatIP with inner product similarity
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2 (768-dim vectors)
- **Retrieval Strategy**: Top-K retrieval + bacterial keyword reranking + full document expansion
- **Generation Model**: DeepSeek API with streaming support
- **Frontend**: Flask + Bootstrap + Markdown rendering
- **Data Processing**: PDF parsing + text chunking + vector indexing

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, for faster embedding)
- 4GB+ RAM
- 2GB+ disk space


## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd amp-chat
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Edit `my_config.py` and set your DeepSeek API key:
```python
DEEPSEEK_API_KEY: str = "your-api-key-here"
```

### 5. Prepare Dataset
Place your PDF research papers in the `datasets/` folder. The system currently includes sample papers about antimicrobial peptides.

## 🎯 Usage

### Command Line Interface
```bash
python main.py
```
This starts an interactive Q&A session in the terminal.

### Web Interface
```bash
cd website
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

### Example Queries
- "What antimicrobial peptides are effective against E. coli?"
- "Show me MIC values for peptides against Staphylococcus aureus"
- "What is the mechanism of action of nisin?"
- "Find peptides with low hemolytic activity"

## 📁 Project Structure

```
amp-chat/
├── main.py                 # Main application entry point
├── my_config.py           # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data_processing/      # PDF loading and text chunking
│   ├── pdf_loader.py
│   └── text_chunker.py
├── models/               # AI models and APIs
│   ├── embedding.py
│   ├── llm.py
│   └── chatapi.py
├── retrieval/            # Vector database and retrieval
│   ├── vector_db.py
│   └── retriever.py
├── generation/           # Response generation
│   ├── generator.py
│   ├── prompt.py
│   └── extractor.py
├── website/              # Web interface
│   ├── app.py
│   ├── requirements.txt
│   └── templates/
│       └── index.html
├── datasets/             # PDF research papers
└── images/               # UI assets
```

## ⚙️ Configuration

Key configuration options in `my_config.py`:

```python
# Device settings
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# API configuration
DEEPSEEK_API_KEY: str = "your-key"
DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"

# Embedding model
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

# Retrieval parameters
SIMILARITY_THRESHOLD = 0.5
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
TOP_K: int = 3
```

## 🔧 Advanced Usage

### Adding New Documents
1. Place PDF files in the `datasets/` folder
2. Restart the application to rebuild the vector index
3. The system will automatically process and index new documents

### Customizing Retrieval
Modify `retrieval/retriever.py` to:
- Add new microorganism keywords
- Adjust similarity thresholds
- Change reranking strategies

### Extending Prompts
Edit `generation/prompt.py` to customize:
- Response formats
- Citation styles
- Output structures

## 🧪 Testing

Run basic functionality tests:
```bash
python -c "from main import AntimicrobialRAG; rag = AntimicrobialRAG('./datasets'); print(rag.query('What is nisin?'))"
```

## 📊 Performance

- **Index Building**: ~30 seconds for 4 research papers
- **Query Response**: ~2-5 seconds per question
- **Memory Usage**: ~2GB RAM with loaded models
- **Accuracy**: Exact vector retrieval (no approximation)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FAISS library for efficient similarity search
- Sentence Transformers for high-quality embeddings
- DeepSeek for language generation capabilities
- Flask for web framework
- Bootstrap for responsive UI design

---

**Note**: This system is designed for research purposes. Always verify critical information from primary sources.
