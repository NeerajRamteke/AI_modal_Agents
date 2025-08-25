import os
import fitz  # PyMuPDF
import hashlib
import re
import base64
from io import BytesIO
from PIL import Image
from typing import List

# --- Web Scraping ---
import requests
from bs4 import BeautifulSoup

# --- Flask ---
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- AI & Data Libraries ---
import google.generativeai as genai
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 🤫 Configuration ---
# API key is kept as requested.
GEMINI_API_KEY = "AIzaSyB_ds-ahs2u6mQtbmYcYNNKTgbwlqR9Iew"

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"🔴 Gemini configuration failed: {e}")
    exit()

# --- 🚀 Flask App Initialization ---
app = Flask(_name_)
# This enables CORS to allow your frontend to communicate with this backend.
CORS(app)

# --- 🧠 Model Initialization ---
# In Flask, we initialize models at the module level.
# This code runs once when the server starts.
print("🚀 Initializing models and database...")
embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}
)

rag_system_prompt = (
    "You are a professional AI research assistant. Provide well-structured, accurate answers based ONLY on the provided context. "
    "When referencing content from a PDF, explicitly mention the page number. If the answer is not in the context, say so."
)
rag_model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=rag_system_prompt)

general_system_prompt = "You are a helpful and friendly AI assistant named Neko."
general_model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=general_system_prompt)

db = chromadb.PersistentClient(path="./chroma_db")
print("✅ Initialization complete.")

# --- 📁 File & Image Helpers ---
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_images_for_pages(doc_path: str, pages: List[int]) -> List[Image.Image]:
    """Extracts PIL images from specific pages of a PDF."""
    images = []
    if not pages:
        return images
        
    unique_pages = sorted(list(set(pages)))
    with fitz.open(doc_path) as doc:
        for page_num in unique_pages:
            if 1 <= page_num <= len(doc):
                page = doc[page_num - 1]
                # Extract all images on the page and add them
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    pil_image = Image.open(BytesIO(base_image["image"]))
                    images.append(pil_image)
    return images

# --- 🤖 Core Logic Functions ---
def process_pdf_and_store(file_contents: bytes, document_id: str):
    """Processes a PDF and stores its content in ChromaDB."""
    collection_name = document_id
    
    try:
        db.get_collection(name=collection_name)
        print(f"✅ Collection '{collection_name}' already exists.")
        return
    except ValueError:
        collection = db.get_or_create_collection(name=collection_name)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    with fitz.open(stream=file_contents, filetype="pdf") as doc:
        docs_to_embed, metadata_to_store, ids_to_store = [], [], []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                chunks = text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    ids_to_store.append(f"page_{page_num}text{i}")
                    docs_to_embed.append(chunk)
                    metadata_to_store.append({"page": page_num, "type": "text"})
        
        if docs_to_embed:
            collection.add(documents=docs_to_embed, metadatas=metadata_to_store, ids=ids_to_store)
    print(f"🌀 Processed and stored PDF document: {document_id}")

def process_url_and_store(url: str, document_id: str):
    """Scrapes a URL, processes its text, and stores it in ChromaDB."""
    collection_name = document_id
    
    try:
        db.get_collection(name=collection_name)
        print(f"✅ Collection for URL '{url}' already exists.")
        return
    except ValueError:
        collection = db.get_or_create_collection(name=collection_name)

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from the body, using a space separator
        text = soup.body.get_text(separator=' ', strip=True)

        if not text:
            print(f"⚠ No text content found at URL: {url}")
            return
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        if chunks:
            ids_to_store = [f"chunk_{i}" for i in range(len(chunks))]
            metadata_to_store = [{"source_url": url, "type": "web_text"} for _ in chunks]
            collection.add(documents=chunks, metadatas=metadata_to_store, ids=ids_to_store)
            print(f"🌀 Processed and stored URL content: {document_id}")

    except requests.exceptions.RequestException as e:
        print(f"🔴 Failed to fetch URL {url}: {e}")
        raise # Re-raise the exception to be caught by the endpoint

# --- 🔗 API Endpoints ---
@app.route("/upload", methods=["POST"])
def upload_pdf():
    """
    Accepts a PDF file, saves it, processes it, and returns a document_id.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.content_type == "application/pdf":
        file_contents = file.read()
        document_id = f"pdf_{hashlib.md5(file_contents).hexdigest()}"
        file_path = os.path.join(UPLOADS_DIR, document_id)

        # Save the file for later image retrieval
        with open(file_path, "wb") as f:
            f.write(file_contents)
            
        process_pdf_and_store(file_contents, document_id)
        
        return jsonify({
            "document_id": document_id, 
            "filename": file.filename
        })
    else:
        return jsonify({"error": "Invalid file type. Only PDF is supported."}), 400

@app.route("/process_url", methods=["POST"])
def process_url():
    """
    Accepts a URL, scrapes it, processes it, and returns a document_id.
    """
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    
    url = data["url"]
    if not re.match(r'^https?://', url):
        return jsonify({"error": "Invalid URL format."}), 400

    try:
        document_id = f"url_{hashlib.md5(url.encode()).hexdigest()}"
        process_url_and_store(url, document_id)
        return jsonify({
            "document_id": document_id,
            "url": url
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process URL: {str(e)}"}), 500

@app.route("/query", methods=["POST"])
def handle_query():
    """
    Handles a user's query. If a document_id is provided, it performs RAG.
    Otherwise, it returns a general knowledge answer.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query = data["query"]
    document_id = data.get("document_id")

    # BEHAVIOR 1: RAG for a specific document (PDF or URL)
    if document_id:
        try:
            collection = db.get_collection(name=document_id)
        except ValueError:
            return jsonify({"error": "Document has not been processed or does not exist."}), 404
            
        results = collection.query(query_texts=[query], n_results=5)
        context_str = "\n---\n".join(results['documents'][0])
        prompt = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}\n\nANSWER:"
        
        response = rag_model.generate_content(prompt)
        answer = response.text
        
        base64_images = []
        # *MODIFIED*: Only try to get images if it's a PDF document
        if document_id.startswith("pdf_"):
            doc_path = os.path.join(UPLOADS_DIR, document_id)
            if not os.path.exists(doc_path):
                print(f"⚠ PDF file not found at {doc_path} for image extraction.")
            else:
                mentioned_pages = set(map(int, re.findall(r'[Pp]age (\d+)', answer)))
                pil_images = get_images_for_pages(doc_path, list(mentioned_pages))
                base64_images = [pil_to_base64(img) for img in pil_images]

        return jsonify({
            "answer": answer,
            "source": "document",
            "images": base64_images
        })
    
    # BEHAVIOR 2: General knowledge question
    else:
        response = general_model.generate_content(query)
        return jsonify({
            "answer": response.text,
            "source": "general_knowledge",
            "images": []
        })

# --- 🏃‍♂ To run this API ---
# In your terminal, use the command: flask --app <filename> run
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=8000, debug=True)
