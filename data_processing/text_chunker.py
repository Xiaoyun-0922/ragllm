from typing import List, Dict
from my_config import config

class TextChunker:
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into chunks"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self._chunk_text(doc["text"])
            
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk,
                    "source": doc["source"],
                    "chunk_id": i,
                    "metadata": doc.get("metadata", {})
                })
        
        return chunked_docs
    
    def _chunk_text(self, text: str) -> List[str]:
        """Text chunking implementation"""
        chunks = []
        start = 0
        end = self.chunk_size
        
        while start < len(text):
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
            end = start + self.chunk_size
        
        return chunks