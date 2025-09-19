from typing import List
from sentence_transformers import SentenceTransformer
import torch
from my_config import config

class EmbeddingModel:
    def __init__(self, device=None):  # Add device parameter
        """Initialize embedding model with device support"""
        self.device = device if device else config.DEVICE  # Use device from config by default
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=self.device  # Pass to SentenceTransformer
        )

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Generate embedding vectors"""
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device  # Ensure using specified device
        )