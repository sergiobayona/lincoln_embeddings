# Abraham Lincoln Letters: Semantic Search Demo Guide

This guide will help you set up and use the semantic search system for exploring Abraham Lincoln's letters using natural language queries.

## Overview

This demo project showcases the power of semantic search to explore historical documents. Unlike traditional keyword search, semantic search understands the meaning behind your questions, allowing you to find relevant content even when the exact words aren't present in the documents.

## Setup Instructions

### 1. Install Required Libraries

Install all dependencies using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

For GPU acceleration (optional but recommended for larger collections):
```bash
pip install faiss-gpu
```

### 2. Lincoln Letters Collection

The Lincoln letters are stored in the base folder called `pdf_documents`. For best results with your own letters:
- Use OCR if you're scanning physical letters
- Maintain consistent formatting where possible
- Name files with descriptive information (e.g., "lincoln_letter_1863_march_15.pdf")

### 3. Run the Demo

The demo can be run in two modes:

#### a. Indexing Mode (First Time Setup)

First, index your collection of Lincoln letters:

```bash
python lincoln_search.py --index --dir ./pdf_documents
```

This will:
- Extract text from all PDFs in the directory
- Attempt to identify metadata like dates and recipients
- Split letters into searchable chunks
- Generate embeddings using Sentence-BERT
- Save the index for future use

#### b. Search Mode

After indexing, you can search in two ways:

**Interactive Mode:**
```bash
python lincoln_search.py
```

This launches an interactive prompt where you can type multiple queries and explore Lincoln's letters.

**Single Query Mode:**
```bash
python lincoln_search.py --query "views on slavery" --results 5 --context
```

## Advanced Usage

### Special Search Commands

When in interactive mode, you can use special commands:

- `/context` - Shows more context around results (e.g., `/context slavery`)
- `/top10` - Shows more results (e.g., `/top10 civil war`)

### Command Line Arguments

```
--index, -i           Index documents (must be done before searching)
--search, -s          Enter interactive search mode
--query, -q           Single search query
--dir, -d             Directory with PDF letters (default: ./pdf_documents)
--model, -m           Sentence-BERT model name (default: all-MiniLM-L6-v2)
--index-file          FAISS index file path (default: lincoln_index.faiss)
--metadata-file       Metadata file path (default: lincoln_metadata.pkl)
--context, -c         Show extended context around results
--results, -r         Number of results to show (default: 5)
```

### Changing the Embedding Model

For improved accuracy (at the cost of speed), try using a more powerful model:

```bash
python lincoln_search.py --index --model all-mpnet-base-v2 --dir ./lincoln_letters
```

## Example Queries

The semantic search understands natural language queries. Try questions like:

- "What did Lincoln think about abolition?"
- "Letters discussing the Civil War strategy"
- "Lincoln's personal feelings about losing his son"
- "Correspondence with military generals"
- "Lincoln's humor and jokes"
- "Views on reconstruction after the war"

## How It Works

1. **Text Extraction**: The system extracts text from PDF files containing Lincoln's letters
2. **Chunking**: Long letters are split into smaller, manageable pieces with overlap
3. **Metadata Extraction**: The system tries to identify dates, recipients, and other metadata
4. **Embedding Generation**: Sentence-BERT converts text into mathematical vectors
5. **Vector Search**: When you search, your query is converted to a vector and compared with the document vectors
6. **Result Ranking**: The most similar documents are returned based on semantic meaning

## Extending the Demo

Here are some ways you can enhance this demo:

1. **Web Interface**: Add a Flask or Streamlit web interface
2. **Enhanced Metadata**: Improve the metadata extraction for dates, locations, and recipients
3. **Timeline Visualization**: Add visualization of search results on a timeline
4. **Network Analysis**: Create relationship graphs between Lincoln and his correspondents
5. **Hybrid Search**: Combine semantic search with keyword search for even better results

## Troubleshooting

- **Memory Issues**: For large collections, reduce batch size or use a smaller model
- **Extraction Problems**: Check PDF quality and OCR accuracy
- **No Results**: Try rephrasing your query or using more general terms
- **Poor Results**: Consider using a more powerful model like "all-mpnet-base-v2"

## Resources

To learn more about the technologies used:

- [Sentence-BERT Documentation](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)
