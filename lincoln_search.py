# lincoln_search.py
import os
import re
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Any
from datetime import datetime
import argparse


class LincolnLettersSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Lincoln Letters semantic search engine.
        
        Args:
            model_name (str): Name of the Sentence-BERT model to use.
        """
        print(f"Initializing search engine with model: {model_name}")
        # Load the Sentence-BERT model
        self.model = SentenceTransformer(model_name)
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index (will be built when documents are added)
        self.index = None
        
        # Storage for document metadata
        self.documents = []
        
        # Track stats
        self.stats = {
            "total_letters": 0,
            "total_chunks": 0,
            "earliest_date": None,
            "latest_date": None
        }
    
    def extract_metadata(self, file_path: str, text: str) -> Dict[str, Any]:
        """
        Extract metadata from Lincoln letter, like date, recipient, etc.
        This is a simplified version - you might want to enhance this with regex patterns
        specific to the format of your Lincoln letters.
        
        Args:
            file_path (str): Path to the PDF file.
            text (str): Extracted text from the PDF.
            
        Returns:
            Dict: Metadata extracted from the letter.
        """
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'letter_date': None,
            'recipient': None,
            'location': None
        }
        
        # Extract date (simple pattern matching - enhance as needed)
        date_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}'
        date_match = re.search(date_pattern, text[:500])  # Look in first 500 chars
        if date_match:
            date_str = date_match.group(0)
            try:
                parsed_date = datetime.strptime(date_str, '%B %d, %Y')
                metadata['letter_date'] = parsed_date
                
                # Update stats
                if self.stats["earliest_date"] is None or parsed_date < self.stats["earliest_date"]:
                    self.stats["earliest_date"] = parsed_date
                if self.stats["latest_date"] is None or parsed_date > self.stats["latest_date"]:
                    self.stats["latest_date"] = parsed_date
            except:
                pass
        
        # Extract recipient (simple pattern - enhance as needed)
        recipient_pattern = r'(To|Dear)\s+([A-Z][a-z]+(\s+[A-Z][a-z]+){0,2})'
        recipient_match = re.search(recipient_pattern, text[:500])
        if recipient_match:
            metadata['recipient'] = recipient_match.group(2)
        
        return metadata
    
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
        
        # Optional: Convert to lowercase
        # text = text.lower()
        
        # Optional: remove specific patterns common in historical letters
        # Add specific cleaning for 19th century letter formatting if needed
        
        return text
    
    def split_into_chunks(self, text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
        """
        Split text into overlapping chunks for better semantic search.
        Using smaller chunks for letters since they tend to be shorter documents.
        
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
    
    def add_letter(self, file_path: str, chunk_size: int = 150, overlap: int = 30) -> None:
        """
        Process a PDF document containing a Lincoln letter and add it to the search engine.
        
        Args:
            file_path (str): Path to the PDF file.
            chunk_size (int): Size of text chunks in words.
            overlap (int): Overlap between chunks in words.
        """
        # Extract and preprocess text
        raw_text = self.extract_text_from_pdf(file_path)
        processed_text = self.preprocess_text(raw_text)
        
        # Extract metadata
        metadata = self.extract_metadata(file_path, raw_text)
        
        # Split into chunks
        chunks = self.split_into_chunks(processed_text, chunk_size, overlap)
        
        # Generate embeddings for chunks
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        # Store document metadata
        doc_id_start = len(self.documents)
        for i, chunk in enumerate(chunks):
            doc_entry = metadata.copy()
            doc_entry.update({
                'id': doc_id_start + i,
                'chunk_index': i,
                'text': chunk
            })
            self.documents.append(doc_entry)
        
        # Update or create the FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.vector_dimension)
        
        # Add embeddings to the index
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)
        
        # Update stats
        self.stats["total_letters"] += 1
        self.stats["total_chunks"] += len(chunks)
    
    def add_letters_from_directory(self, directory_path: str, chunk_size: int = 150, overlap: int = 30) -> None:
        """
        Process all PDF documents in a directory and add them to the search engine.
        
        Args:
            directory_path (str): Path to the directory containing PDF files.
            chunk_size (int): Size of text chunks in words.
            overlap (int): Overlap between chunks in words.
        """
        print(f"Processing Lincoln letters from: {directory_path}")
        letter_count = 0
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                print(f"  Processing letter: {filename}")
                self.add_letter(file_path, chunk_size, overlap)
                letter_count += 1
        
        print(f"Added {letter_count} letters with {self.stats['total_chunks']} total chunks")
        
        if self.stats["earliest_date"] and self.stats["latest_date"]:
            print(f"Letters span from {self.stats['earliest_date'].strftime('%B %d, %Y')} to {self.stats['latest_date'].strftime('%B %d, %Y')}")
    
    def search(self, query: str, top_k: int = 5, filter_date_start=None, filter_date_end=None, filter_recipient=None) -> List[Dict[str, Any]]:
        """
        Search for letters semantically similar to the query with optional filters.
        
        Args:
            query (str): Search query.
            top_k (int): Number of results to return.
            filter_date_start: Optional start date filter
            filter_date_end: Optional end date filter
            filter_recipient: Optional recipient filter
            
        Returns:
            List[Dict]: List of search results with metadata.
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Search the index - get more results than needed to allow for filtering
        k_multiplier = 3 if (filter_date_start or filter_date_end or filter_recipient) else 1
        distances, indices = self.index.search(query_embedding, top_k * k_multiplier)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure index is valid
                result = self.documents[idx].copy()
                result['score'] = 1 - distances[0][i]  # Convert distance to similarity score
                
                # Apply filters
                include = True
                if filter_date_start and result['letter_date'] and result['letter_date'] < filter_date_start:
                    include = False
                if filter_date_end and result['letter_date'] and result['letter_date'] > filter_date_end:
                    include = False
                if filter_recipient and result['recipient'] and filter_recipient.lower() not in result['recipient'].lower():
                    include = False
                
                if include:
                    results.append(result)
        
        # Sort by score and limit to top_k
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return results
    
    def get_context(self, result, context_size=2):
        """
        Get surrounding chunks for more context around a search result.
        
        Args:
            result: The search result
            context_size: Number of chunks to include before and after
            
        Returns:
            str: Extended context
        """
        file_path = result['file_path']
        chunk_idx = result['chunk_index']
        
        # Find all chunks from the same document
        same_doc_chunks = [d for d in self.documents if d['file_path'] == file_path]
        same_doc_chunks.sort(key=lambda x: x['chunk_index'])
        
        # Find the position of this chunk
        try:
            pos = next(i for i, chunk in enumerate(same_doc_chunks) if chunk['chunk_index'] == chunk_idx)
        except StopIteration:
            return result['text']
        
        # Get context chunks
        start = max(0, pos - context_size)
        end = min(len(same_doc_chunks), pos + context_size + 1)
        context_chunks = same_doc_chunks[start:end]
        
        # Combine text
        return " ".join([c['text'] for c in context_chunks])
    
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
            
            # Save stats
            pd.Series(self.stats).to_pickle(metadata_path + ".stats")
            
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
        
        # Load stats if available
        try:
            self.stats = pd.read_pickle(metadata_path + ".stats").to_dict()
        except:
            # Regenerate stats
            self.stats = {
                "total_letters": len(set([d['file_path'] for d in self.documents])),
                "total_chunks": len(self.documents),
                "earliest_date": None,
                "latest_date": None
            }
            
            # Recalculate date range
            dates = [d['letter_date'] for d in self.documents if d['letter_date'] is not None]
            if dates:
                self.stats["earliest_date"] = min(dates)
                self.stats["latest_date"] = max(dates)
        
        print(f"Loaded index with {self.stats['total_chunks']} chunks from {self.stats['total_letters']} Lincoln letters")
        if self.stats["earliest_date"] and self.stats["latest_date"]:
            print(f"Letters span from {self.stats['earliest_date'].strftime('%B %d, %Y')} to {self.stats['latest_date'].strftime('%B %d, %Y')}")


def format_search_results(results, search_engine, show_context=False):
    """Format search results for display"""
    output = []
    for i, result in enumerate(results):
        output.append(f"\n=== Result {i+1} (Score: {result['score']:.4f}) ===")
        
        # Add metadata if available
        if result['letter_date']:
            output.append(f"Date: {result['letter_date'].strftime('%B %d, %Y')}")
        if result['recipient']:
            output.append(f"To: {result['recipient']}")
            
        output.append(f"Source: {result['file_name']}")
        
        # Add text content
        if show_context:
            output.append("\nEXTENDED CONTEXT:")
            output.append(search_engine.get_context(result))
        else:
            output.append("\nEXCERPT:")
            output.append(result['text'])
            
        output.append("-" * 60)
    
    return "\n".join(output)


def interactive_search(search_engine):
    """Run an interactive search session"""
    print("\n" + "=" * 80)
    print("   ABRAHAM LINCOLN LETTERS SEARCH ENGINE")
    print("=" * 80)
    print(f"Loaded {search_engine.stats['total_letters']} letters with {search_engine.stats['total_chunks']} searchable chunks")
    
    if search_engine.stats["earliest_date"] and search_engine.stats["latest_date"]:
        print(f"Letters span from {search_engine.stats['earliest_date'].strftime('%B %d, %Y')} to {search_engine.stats['latest_date'].strftime('%B %d, %Y')}")
    
    print("\nEnter search queries (or 'quit' to exit):")
    
    while True:
        query = input("\nSearch query: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        # Parse for command options
        show_context = False
        num_results = 5
        
        if query.startswith("/"):
            parts = query.split(" ", 1)
            if len(parts) == 2:
                command, query = parts
                if command == "/context":
                    show_context = True
                elif command.startswith("/top"):
                    try:
                        num_results = int(command[4:])
                    except:
                        pass
        
        # Execute search
        results = search_engine.search(query, top_k=num_results)
        
        if not results:
            print("No results found.")
            continue
            
        # Display results
        print(format_search_results(results, search_engine, show_context))


def main():
    parser = argparse.ArgumentParser(description='Lincoln Letters Semantic Search Engine')
    parser.add_argument('--index', '-i', action='store_true', help='Index documents')
    parser.add_argument('--search', '-s', action='store_true', help='Search mode')
    parser.add_argument('--query', '-q', type=str, help='Search query')
    parser.add_argument('--dir', '-d', type=str, default='./pdf_documents', help='Directory with PDF letters')
    parser.add_argument('--model', '-m', type=str, default='all-MiniLM-L6-v2', help='Sentence-BERT model name')
    parser.add_argument('--index-file', type=str, default='lincoln_index.faiss', help='FAISS index file')
    parser.add_argument('--metadata-file', type=str, default='lincoln_metadata.pkl', help='Metadata file')
    parser.add_argument('--context', '-c', action='store_true', help='Show extended context')
    parser.add_argument('--results', '-r', type=int, default=5, help='Number of results to show')
    
    args = parser.parse_args()
    
    # Initialize the search engine
    search_engine = LincolnLettersSearch(model_name=args.model)
    
    # Index or load
    if args.index:
        search_engine.add_letters_from_directory(args.dir)
        search_engine.save_index(args.index_file, args.metadata_file)
    else:
        try:
            search_engine.load_index(args.index_file, args.metadata_file)
        except Exception as e:
            print(f"Error loading index: {e}")
            print("You may need to index documents first with --index")
            return
    
    # Search modes
    if args.query:
        results = search_engine.search(args.query, top_k=args.results)
        print(format_search_results(results, search_engine, args.context))
    elif args.search or not args.index:
        interactive_search(search_engine)


if __name__ == "__main__":
    main()