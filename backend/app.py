from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from werkzeug.utils import secure_filename
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import traceback
import json
import fitz  # PyMuPDF
from io import BytesIO
import base64
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
GEMINI_API_KEY = "AIzaSyB_ds-ahs2u6mQtbmYcYNNKTgbwlqR9Iew"
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
CHROMA_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db')

# Initialize ChromaDB
os.makedirs(CHROMA_DIR, exist_ok=True)
try:
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )
    print(f"✅ ChromaDB initialized at {os.path.abspath(CHROMA_DIR)}")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {str(e)}")
    raise

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize text splitter with better settings for web content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Increased from 1000 to 2000 for better context
    chunk_overlap=400,  # Increased from 200 to 400 for better context preservation
    length_function=len,
    separators=["\n\n", "\n", " ", ""],  # Explicit separators for better chunking
    is_separator_regex=False,
)

# Document store
document_store = {
    'processed_files': {}
}

# Helper functions
def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

@app.route('/api/load-url', methods=['POST'])
def load_url():
    """Load content from URL for RAG operations"""
    try:
        data = request.get_json()
        url = data.get('url')
        include_images = data.get('include_images', False)
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
            
        print(f"🌐 Loading content from URL: {url}")
        
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return jsonify({"error": "Invalid URL format"}), 400
        except Exception as e:
            return jsonify({"error": f"Invalid URL: {str(e)}"}), 400
        
        # Fetch URL content with error handling
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return jsonify({"error": f"URL does not contain HTML content (content-type: {content_type})"}), 400
                
        except requests.RequestException as e:
            return jsonify({"error": f"Failed to fetch URL: {str(e)}"}), 400
        
        # Parse HTML content with BeautifulSoup
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript', 'svg', 'button', 'form', 'aside', 'nav', 'menu', 'dialog', 'figure']):
                element.decompose()
            
            # Extract main content - try to find the main article content
            article_selectors = [
                'article',
                'main',
                'div[role="main"]',
                'div.content',
                'div#content',
                'div.main',
                'div#main',
                'div.container',
                'div.wrapper',
                'body'
            ]
            
            content = None
            for selector in article_selectors:
                element = soup.select_one(selector)
                if element and len(element.get_text(strip=True)) > 500:  # Only use if it has substantial content
                    content = element
                    print(f"✅ Found content using selector: {selector}")
                    break
            
            if not content:
                content = soup.find('body') or soup
            
            # Process paragraphs to maintain structure
            paragraphs = []
            for p in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                text = p.get_text('\n', strip=True)
                if len(text) > 10:  # Only include non-empty paragraphs
                    # Add extra newlines around headings for better chunking
                    if p.name.startswith('h'):
                        paragraphs.append(f'\n\n{text.upper()}\n')
                    else:
                        paragraphs.append(text)
            
            # Join with double newlines to preserve paragraph structure
            content = '\n\n'.join(paragraphs)
            
            if not content.strip():
                return jsonify({"error": "No text content found on the page"}), 400
                
            print(f"✅ Extracted {len(content)} characters from URL")
            
        except Exception as e:
            return jsonify({"error": f"Error parsing HTML content: {str(e)}"}), 500
        
        # Create document ID
        doc_id = str(uuid.uuid4())
        
        # Split content into chunks
        chunks = text_splitter.split_text(content)
        
        # Prepare documents for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"url_{doc_id}_{i}"
            
            # Prepare metadata with serializable values
            metadata = {
                "source": str(url),
                "title": soup.title.string if soup.title else "No title",
                "chunk": int(i),
                "has_images": False,  # No image support from URLs yet
                "page_numbers": "[1]",  # Store as string
                "type": "webpage",
                "url": str(url),
                "processed_at": datetime.now().isoformat()
            }
            
            # Ensure all metadata values are serializable
            for k, v in list(metadata.items()):
                if v is None or isinstance(v, (str, int, float, bool)):
                    continue
                metadata[k] = str(v)
            
            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        print(f"📊 Adding {len(documents)} chunks from URL to ChromaDB...")
        
        # Add to ChromaDB
        if documents:
            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print("✅ Successfully added URL content to ChromaDB")
            except Exception as e:
                print(f"❌ Error adding to ChromaDB: {str(e)}")
                return jsonify({"error": f"Failed to add to database: {str(e)}"}), 500
        
        # Store URL info in document store
        document_store['processed_files'][doc_id] = {
            'filename': url,
            'title': soup.title.string if soup.title else "No title",
            'chunks': len(chunks),
            'processed_at': datetime.now().isoformat(),
            'type': 'webpage',
            'url': url,
            'content_length': len(content)
        }
        
        return jsonify({
            "document_id": doc_id,
            "title": soup.title.string if soup.title else "No title",
            "url": url,
            "chunks_processed": len(chunks),
            "content_length": len(content),
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ Error in load_url: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error processing URL: {str(e)}"}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint for querying documents"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        document_id = data.get('document_id')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        print(f"🔍 Processing search query: {query}")
        if document_id:
            print(f"🔎 Filtering by document_id: {document_id}")
        
        # Generate query embedding
        try:
            query_embedding = embedding_model.embed_query(query)
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": 5,
                "include": ["documents", "metadatas", "distances"],
            }
            
            # Add document filter if document_id is provided
            if document_id:
                query_params["where"] = {"$or": [
                    {"document_id": document_id},
                    {"url": document_id}  # Also match by URL for backward compatibility
                ]}
            
            # Execute the query
            results = collection.query(**query_params)
            
            # Process results
            search_results = []
            
            if results.get('documents') and results['documents'][0]:
                for i, (doc, metadata, score) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Ensure metadata is a dictionary
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    # Skip results with very low similarity
                    similarity = 1.0 - (score / 2.0)  # Convert distance to similarity (0-1)
                    if similarity < 0.2:  # Skip very low similarity results
                        continue
                    
                    # Prepare the result with enhanced metadata
                    result = {
                        'text': doc,
                        'metadata': {
                            'source': metadata.get('source', 'Unknown'),
                            'title': metadata.get('title', 'Untitled'),
                            'url': metadata.get('url', ''),
                            'chunk': metadata.get('chunk', 0),
                            'document_id': metadata.get('document_id', document_id),
                            'type': metadata.get('type', 'webpage'),
                            'score': round(similarity, 4)
                        },
                        'score': similarity
                    }
                    
                    search_results.append(result)
            
            # Sort by score (highest first)
            search_results.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"🔎 Found {len(search_results)} relevant results")
            
            return jsonify({
                'query': query,
                'results': search_results,
                'total_results': len(search_results),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Search failed: {str(e)}"}), 500
            
    except Exception as e:
        print(f"❌ Error in search: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error processing search: {str(e)}"}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(os.path.join(UPLOADS_DIR, 'images'), exist_ok=True)
    
    print("🚀 Starting AstraFind Backend Server...")
    print("📡 Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
