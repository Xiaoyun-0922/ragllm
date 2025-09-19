from pypdf import PdfReader
from typing import List, Dict
import os

class PDFLoader:
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
    
    def load_pdfs(self) -> List[Dict]:
        """Load all PDF files from the folder"""
        documents = []
        
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith('.pdf'):
                filepath = os.path.join(self.pdf_folder, filename)
                try:
                    text = self._extract_text_from_pdf(filepath)
                    documents.append({
                        "text": text,
                        "source": filename,
                        "metadata": {"filepath": filepath}
                    })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return documents
    
    def _extract_text_from_pdf(self, filepath: str) -> str:
        """Extract text from a single PDF file"""
        reader = PdfReader(filepath)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text