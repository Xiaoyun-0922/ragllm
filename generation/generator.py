import os
from typing import Dict, List
from models.llm import DeepSeekAPI  # Replace original LLM
from generation.prompt import PromptBuilder

class ResponseGenerator:
    def __init__(self):
        self.llm = DeepSeekAPI()  # Use DeepSeek API
        self.prompt_builder = PromptBuilder()

    def generate(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer for antimicrobial peptide (AMP) questions."""
        context_texts = [doc["text"] for doc in context_docs]
        # 1. extract key info
        from generation.extractor import extract_key_info
        seq, mic, struct = extract_key_info(context_texts)
        info_lines = []
        if seq:
            info_lines.append(f"AMP Sequence: {seq}")
        if mic:
            info_lines.append(f"MIC: {mic}")
        if struct:
            info_lines.append(f"Structure: {struct}")
        info_str = "\n".join(info_lines)
        context = "\n\n".join(context_texts)
        # 2. prepend source list so the model can cite actual filenames
        sources = sorted({os.path.basename(doc.get("source", "")) for doc in context_docs if doc.get("source")})
        sources_line = f"Sources: {'; '.join(sources)}" if sources else ""
        # 3. construct prompt
        parts = [sources_line]
        if info_str:
            parts.append(info_str)
        parts.append(context)
        context_for_prompt = "\n\n".join([p for p in parts if p])
        prompt = self.prompt_builder.build_rag_prompt_amp_answer(query, context_for_prompt)
        # 4. use DeepSeek API
        response = self.llm.generate(prompt)
        return response
    def generate_modeling_report(self, context_docs: List[Dict], query: str) -> str:
        """Generate Section 8 mathematical modeling report."""
        # Build context (including system architecture information)
        sources = sorted({os.path.basename(doc.get("source", "")) for doc in context_docs if doc.get("source")})
        sources_line = f"Sources: {'; '.join(sources)}" if sources else ""

        # System architecture information (extracted from codebase)
        system_info = """
System Architecture Overview:
- Vector Database: FAISS IndexFlatIP, supports inner product similarity computation
- Embedding Model: sentence-transformers/all-mpnet-base-v2, 768-dimensional vectors
- Retrieval Strategy: Top-K retrieval + bacterial keyword reranking + full document expansion
- Generation Model: DeepSeek API, supports streaming generation
- Frontend: Flask + Bootstrap + Markdown rendering
- Data Processing: PDF parsing + text chunking + vector indexing
"""

        context = "\n\n".join([p for p in [sources_line, system_info] if p])

        prompt = PromptBuilder.build_modeling_report_prompt(context, query)
        return self.llm.generate(prompt)