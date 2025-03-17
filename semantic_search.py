import os
import re
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Any


class SemanticSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic search engine with a Sentence-BERT model.
        
        Args:
            model_name (str): Name of the Sentence-BERT model to use.
        """
        # Load the Sentence-BERT model
        self.model = SentenceTransformer(model_name)
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (will be built when documents are added)
        self.index = None
        
        # Storage for document metadata
        self.documents = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            str: Extracted text from the PDF.
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the extracted text.
        
        Args:
            text (str): Raw text from PDF.
            
        Returns:
            str: Preprocessed text.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters if needed
        # text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def split_into_chunks(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better semantic search.
        
        Args:
            text (str): Text to split.
            chunk_size (int): Approximate chunk size in words.
            overlap (int): Number of overlapping words between chunks.
            
        Returns:
            List[str]: List of text chunks.
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
                
        return chunks
    
    def add_document(self, file_path: str, chunk_size: int = 200, overlap: int = 50) -> None:
        """
        Process a PDF document and add it to the search engine.
        
        Args:
            file_path (str): Path to the PDF file.
            chunk_size (int): Size of text chunks in words.
            overlap (int): Overlap between chunks in words.
        """
        # Extract and preprocess text
        raw_text = self.extract_text_from_pdf(file_path)
        processed_text = self.preprocess_text(raw_text)
        
        # Split into chunks
        chunks = self.split_into_chunks(processed_text, chunk_size, overlap)
        
        # Generate embeddings for chunks
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        # Store document metadata
        doc_id_start = len(self.documents)
        for i, chunk in enumerate(chunks):
            self.documents.append({
                'id': doc_id_start + i,
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'chunk_index': i,
                'text': chunk
            })
        
        # Update or create the FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.vector_dimension)
        
        # Add embeddings to the index
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)
    
    def add_documents_from_directory(self, directory_path: str, chunk_size: int = 200, overlap: int = 50) -> None:
        """
        Process all PDF documents in a directory and add them to the search engine.
        
        Args:
            directory_path (str): Path to the directory containing PDF files.
            chunk_size (int): Size of text chunks in words.
            overlap (int): Overlap between chunks in words.
        """
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                print(f"Processing {filename}...")
                self.add_document(file_path, chunk_size, overlap)
        
        print(f"Added {len(self.documents)} chunks from {directory_path}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents semantically similar to the query.
        
        Args:
            query (str): Search query.
            top_k (int): Number of results to return.
            
        Returns:
            List[Dict]: List of search results with metadata.
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure index is valid
                result = self.documents[idx].copy()
                result['score'] = 1 - distances[0][i]  # Convert distance to similarity score
                results.append(result)
        
        return results
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and document metadata.
        
        Args:
            index_path (str): Path to save the FAISS index.
            metadata_path (str): Path to save document metadata.
        """
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save document metadata
            pd.DataFrame(self.documents).to_pickle(metadata_path)
            
            print(f"Index saved to {index_path}")
            print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """
        Load a saved FAISS index and document metadata.
        
        Args:
            index_path (str): Path to the saved FAISS index.
            metadata_path (str): Path to the saved document metadata.
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load document metadata
        df = pd.read_pickle(metadata_path)
        self.documents = df.to_dict('records')
        
        print(f"Loaded index with {len(self.documents)} documents")


# Example usage
if __name__ == "__main__":
    # Initialize the search engine
    search_engine = SemanticSearchEngine()
    
    # Add documents from a directory
    search_engine.add_documents_from_directory("./pdf_documents")
    
    # Save the index
    search_engine.save_index("search_index.faiss", "document_metadata.pkl")
    
    # Or load a previously saved index
    # search_engine.load_index("search_index.faiss", "document_metadata.pkl")
    
    # Perform a search
    results = search_engine.search("What is machine learning?", top_k=3)
    
    # Print results
    for result in results:
        print(f"Document: {result['file_name']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:150]}...")
        print("-" * 50)