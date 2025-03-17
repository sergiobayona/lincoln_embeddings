from semantic_search import SemanticSearchEngine

# Initialize the search engine
search_engine = SemanticSearchEngine()

# Index your documents
search_engine.add_documents_from_directory("./pdf_documents")

# Save the index for future use
search_engine.save_index("search_index.faiss", "document_metadata.pkl")

# Search for information
results = search_engine.search("Who was McClellan?", top_k=5)

# Display results
for result in results:
    print(f"Document: {result['file_name']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text'][:150]}...")
    print("-" * 50)