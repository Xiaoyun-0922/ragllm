import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from data_processing.pdf_loader import PDFLoader
from data_processing.text_chunker import TextChunker
from models.embedding import EmbeddingModel
from retrieval.vector_db import VectorDB
from retrieval.retriever import Retriever
from generation.generator import ResponseGenerator

from my_config import config
import numpy as np
import os
import pickle
import faiss

class AntimicrobialRAG:
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        self._initialize_components()

    def _initialize_components(self):
        """Initialize components (no disk cache; encode and build index on each startup)."""
        # 1) Load PDFs and chunk documents
        pdf_loader = PDFLoader(self.pdf_folder)
        documents = pdf_loader.load_pdfs()
        chunker = TextChunker()
        self.chunked_documents = chunker.chunk_documents(documents)

        # 2) Encode all chunk texts
        self.embedding_model = EmbeddingModel()
        embeddings = self.embedding_model.encode(
            [doc["text"] for doc in self.chunked_documents]
        ).cpu().numpy()
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # 3) Build vector index (memory only, no persistence)
        self.vector_db = VectorDB(embeddings.shape[1])
        self.vector_db.add_documents(embeddings, self.chunked_documents)

        # Initialize other components
        self.embedding_model = EmbeddingModel()
        self.retriever = Retriever(self.vector_db, self.embedding_model)
        self.generator = ResponseGenerator()

    def _expand_to_full_document(self, relevant_docs):
        """Expand to include all chunks from the top-scoring document source.
        This ensures the LLM sees the full paper when the user asks about a specific microbe.
        """
        if not relevant_docs:
            return relevant_docs
        top_source = relevant_docs[0].get("source")
        if not top_source:
            return relevant_docs
        # Collect all chunks from the same source and sort by chunk_id for natural order
        full_docs = [d for d in self.vector_db.documents if d.get("source") == top_source]
        full_docs.sort(key=lambda d: d.get("chunk_id", 0))
        return full_docs



    def _is_amp_related_query(self, question: str) -> bool:
        """Check if the query is related to antimicrobial peptides"""
        question_lower = question.lower()

        # Simple greetings and casual conversation
        casual_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'what\'s up', 'thanks', 'thank you', 'bye', 'goodbye',
            'test', 'testing', '你好', '谢谢', '再见'
        ]

        # Non-AMP topics that should be filtered out
        non_amp_topics = [
            'machine learning', 'deep learning', 'artificial intelligence', 'ai',
            'programming', 'python', 'javascript', 'coding', 'software',
            'weather', 'news', 'politics', 'sports', 'music', 'movie',
            'recipe', 'cooking', 'travel', 'history', 'geography',
            'math', 'physics', 'chemistry' # unless specifically about AMP chemistry
        ]

        # If it's just a casual greeting, return False
        if any(pattern in question_lower for pattern in casual_patterns) and len(question.strip()) < 30:
            return False

        # If it contains non-AMP topics, return False
        if any(topic in question_lower for topic in non_amp_topics):
            # Exception: if it also contains AMP keywords, it might be AMP-related
            amp_keywords = [
                'antimicrobial', 'peptide', 'amp', 'mic', 'bacteria', 'bacterial',
                'staphylococcus', 'aureus', 'coli', 'pseudomonas', 'hemolysis',
                'sequence', 'mechanism', 'toxicity', 'gram-positive', 'gram-negative',
                '抗菌肽', '细菌', '序列', '机理'
            ]
            if not any(keyword in question_lower for keyword in amp_keywords):
                return False

        # AMP-related keywords
        amp_keywords = [
            'antimicrobial', 'peptide', 'amp', 'mic', 'bacteria', 'bacterial',
            'staphylococcus', 'aureus', 'coli', 'pseudomonas', 'hemolysis',
            'sequence', 'mechanism', 'toxicity', 'gram-positive', 'gram-negative',
            'nisin', 'lysozyme', 'defensin', 'magainin', 'cecropin',
            '抗菌肽', '细菌', '序列', '机理'
        ]

        # If contains AMP keywords, it's likely AMP-related
        if any(keyword in question_lower for keyword in amp_keywords):
            return True

        # For longer questions, be more conservative - only assume AMP-related if it's very likely
        if len(question.strip()) > 15:
            # Look for scientific/biological context that might be AMP-related
            bio_indicators = ['protein', 'amino acid', 'bioactive', 'therapeutic', 'antibiotic']
            if any(indicator in question_lower for indicator in bio_indicators):
                return True

        return False

    def query(self, question: str) -> str:
        """Process a single query and return the answer"""
        # Check if this is an AMP-related query
        if not self._is_amp_related_query(question):
            return "Hello! I'm an AI assistant specialized in antimicrobial peptide research. Please ask me questions about antimicrobial peptides, their sequences, MIC values, mechanisms of action, or related topics."

        relevant_docs = self.retriever.retrieve(question)
        full_docs = self._expand_to_full_document(relevant_docs)
        response = self.generator.generate(question, full_docs)
        return response

    def stream_query(self, question: str):
        """Stream answer generation, word by word output"""
        # Check if this is an AMP-related query
        if not self._is_amp_related_query(question):
            greeting_response = "Hello! I'm an AI assistant specialized in antimicrobial peptide research. Please ask me questions about antimicrobial peptides, their sequences, MIC values, mechanisms of action, or related topics."
            for char in greeting_response:
                yield char
            return

        relevant_docs = self.retriever.retrieve(question)
        full_docs = self._expand_to_full_document(relevant_docs)
        from generation.prompt import PromptBuilder
        # Prepend source list so the model can cite actual filenames
        import os as _os
        _sources = sorted({_os.path.basename(d.get("source", "")) for d in full_docs if d.get("source")})
        _sources_line = f"Sources: {'; '.join(_sources)}" if _sources else ""
        _context_body = "\n\n".join([doc["text"] for doc in full_docs])
        context = "\n\n".join([p for p in [_sources_line, _context_body] if p])
        prompt = PromptBuilder.build_rag_prompt_amp_answer(question, context)
        from my_config import config
        import requests
        url = config.DEEPSEEK_API_URL
        headers = {
            "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 800,
            "top_p": 0.9,
            "stream": True
        }
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    import json as _json
                    try:
                        delta = _json.loads(data)
                        content = delta["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

    def chat(self):
        print("Antimicrobial Peptide Q&A System started. Type 'quit' or 'exit' to end conversation.")
        print("="*50)

        while True:
            query = input("\nUser: ")

            if query.lower() in ['quit', 'exit']:
                print("System: Goodbye!")
                break

            response = self.query(query)
            print(f"\nSystem: {response}")



if __name__ == "__main__":
    PDF_FOLDER = "./datasets"
    rag_system = AntimicrobialRAG(PDF_FOLDER)
    print("Antimicrobial Peptide Q&A System started. Type 'quit' or 'exit' to end conversation.\n" + "="*50)
    while True:
        query = input("\nUser: ")
        if query.lower() in ['quit', 'exit']:
            print("System: Goodbye!")
            break
        print("\nSystem: ", end="", flush=True)
        for chunk in rag_system.stream_query(query):
            print(chunk, end="", flush=True)
        print()