from dataclasses import dataclass
import torch

@dataclass
class config: 
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    DEEPSEEK_API_KEY: str = "..." # Your deepseek API key.
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DEVICE: str = "cuda"

    SIMILARITY_THRESHOLD = 0.5
    MAX_CONTEXT_LENGTH = 20000
    CHUNK_SIZE: int = 500  
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3

config = config() 