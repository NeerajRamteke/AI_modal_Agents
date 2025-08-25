from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from datetime import datetime
import uuid
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import base64
import io
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global storage for processed documents
document_store = {
    'texts': [],
    'metadata': [],
    'processed_files': {}
}

def extract_content_from_pdf(pdf_path):
    """Extract text and images from PDF file using PyMuPDF"""
    content = {
        'text': '',
        'images': [],
        'pages': []
    }
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text
            page_text = page.get_text()
            content['text'] += page_text + "\n"
            
            # Extract images
            image_list = page.get_images()
            page_images = []
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode()
                        
                        # Save image to uploads directory
                        img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page{page_num + 1}_img{img_index}.png"
                        img_path = os.path.join(os.path.dirname(pdf_path), 'images', img_filename)
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        
                        with open(img_path, 'wb') as img_file:
                            img_file.write(img_data)
                        
                        page_images.append({
                            'page': page_num + 1,
                            'index': img_index,
                            'data': f"data:image/png;base64,{img_base64}",
                            'url': f"/api/image/{img_filename}",
                            'filename': img_filename,
                            'width': pix.width,
                            'height': pix.height
                        })
                    
                    pix = None
                except Exception as e:
                    print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
            
            content['images'].extend(page_images)
            content['pages'].append({
                'page_num': page_num + 1,
                'text': page_text,
                'images': page_images
            })
        
        doc.close()
        
    except Exception as e:
        print(f"Error extracting content from PDF: {e}")
    
    return content

def chunk_content(content, chunk_size=500, overlap=50):
    """Split content into overlapping chunks with associated images"""
    text = content['text']
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_text = ' '.join(words[i:i + chunk_size])
        if chunk_text.strip():
            # Find relevant images for this chunk
            chunk_images = []
            chunk_start_pos = text.find(chunk_text)
            
            # Simple heuristic: include images from pages that contain this text
            for page in content['pages']:
                if any(word in page['text'].lower() for word in chunk_text.lower().split()[:10]):
                    chunk_images.extend(page['images'])
            
            chunks.append({
                'text': chunk_text,
                'images': chunk_images[:3],  # Limit to 3 images per chunk
                'chunk_id': len(chunks)
            })
    
    return chunks

def generate_chatgpt_style_response(query, chunks):
    """Generate ChatGPT-style conversational response with inline images"""
    try:
        # Combine all relevant text chunks
        combined_text = "\n\n".join([chunk['text'] for chunk in chunks])
        
        # Collect all images from chunks with context
        all_images = []
        image_contexts = []
        for i, chunk in enumerate(chunks):
            chunk_images = chunk.get('images', [])
            for img in chunk_images:
                all_images.append(img)
                # Add context about where this image came from
                img_context = f"This image appears in section {i+1} which discusses: {chunk['text'][:100]}..."
                image_contexts.append(img_context)
        
        # Enhanced content extraction based on query
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Check if query is too generic
        generic_terms = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'a', 'an'}
        broad_terms = {'science', 'engineering', 'technology', 'research', 'study', 'field', 'area', 'topic', 'subject'}
        meaningful_words = query_words - generic_terms
        
        # Check for overly generic queries
        is_too_generic = (
            len(meaningful_words) <= 1 or  # Very few meaningful words
            all(len(word) <= 3 for word in meaningful_words) or  # All short words
            all(word in broad_terms for word in meaningful_words) or  # All broad academic terms
            (len(meaningful_words) == 2 and any(word in broad_terms for word in meaningful_words))  # Two words with one being broad
        )
        
        if is_too_generic:
            # Handle very generic queries
            return {
                'response': f"Your query '{query}' is quite broad. Could you be more specific? For example:\n\n• What specific aspect of {' '.join(meaningful_words)} are you interested in?\n• Are you looking for definitions, processes, examples, or applications?\n• Try adding more specific terms to your search.\n\nBased on your uploaded document, I can help with topics related to: Artificial Intelligence, Machine Learning, and related technical concepts.",
                'images': all_images[:3],  # Show some images as examples
                'image_contexts': image_contexts[:3],
                'source_chunks': len(chunks),
                'total_images': len(all_images),
                'style': 'conversational',
                'relevance_score': 0.3  # Low score for generic queries
            }
        
        # Find the most relevant sentences with stricter matching
        all_sentences = []
        for chunk in chunks:
            text = chunk['text']
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Score each sentence for relevance with stricter criteria
                score = 0
                
                # Exact phrase match (highest weight)
                if query_lower in sentence_lower:
                    score += 5
                
                # Require at least 2 meaningful words to match
                sentence_words = set(sentence_lower.split())
                common_meaningful_words = meaningful_words.intersection(sentence_words)
                
                if len(common_meaningful_words) >= 2:
                    # Strong match - multiple meaningful words
                    score += len(common_meaningful_words) / len(meaningful_words) * 3
                elif len(common_meaningful_words) == 1:
                    # Weak match - only one meaningful word
                    score += 0.5
                
                # Contextual relevance - words appearing close together
                if len(common_meaningful_words) > 1:
                    for word1 in common_meaningful_words:
                        for word2 in common_meaningful_words:
                            if word1 != word2:
                                pos1 = sentence_lower.find(word1)
                                pos2 = sentence_lower.find(word2)
                                if pos1 != -1 and pos2 != -1 and abs(pos1 - pos2) < 50:
                                    score += 1
                
                # Only include sentences with meaningful relevance
                if score > 1.0:  # Higher threshold for better precision
                    all_sentences.append((sentence, score))
        
        # Sort sentences by relevance
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if not all_sentences:
            # No relevant content found - provide better guidance
            return {
                'response': f"I couldn't find specific information about '{query}' in your uploaded documents.\n\nYour document appears to focus on Artificial Intelligence topics. The content includes information about:\n• AI definitions and concepts\n• Machine learning fundamentals\n• Technical processes and methods\n• Related algorithms and applications\n\nIf you're looking for team or organizational information, this document may not contain that type of content. Try searching for technical topics that match the document's focus, or upload a different document that contains the information you're seeking.",
                'images': all_images[:2],
                'image_contexts': image_contexts[:2],
                'source_chunks': len(chunks),
                'total_images': len(all_images),
                'style': 'conversational',
                'relevance_score': 0.15  # Lower score for no matches
            }
        
        top_sentences = [s[0] for s in all_sentences[:6]]  # Get top 6 most relevant sentences
        
        # Generate detailed response
        response_parts = []
        
        # Dynamic opening based on query and content found
        if any(word in query_lower for word in ['what is', 'define', 'explain']):
            response_parts.append(f"Here's what I found about {query}:")
        elif any(word in query_lower for word in ['how', 'process', 'work']):
            response_parts.append(f"Here's information about {query}:")
        else:
            response_parts.append(f"Based on your documents, here's what I found about '{query}':")
        
        response_parts.append("")
        
        # Main content with organized information
        if top_sentences:
            # Group sentences by topic/theme for better organization
            key_points = []
            definitions = []
            processes = []
            examples = []
            
            for sentence in top_sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in ['is defined', 'refers to', 'means', 'is a', 'definition']):
                    definitions.append(sentence)
                elif any(word in sentence_lower for word in ['process', 'step', 'method', 'approach', 'procedure']):
                    processes.append(sentence)
                elif any(word in sentence_lower for word in ['example', 'such as', 'including', 'like', 'instance']):
                    examples.append(sentence)
                else:
                    key_points.append(sentence)
            
            # Present information in organized sections
            if definitions:
                response_parts.append("**Definition & Overview:**")
                for def_sentence in definitions[:2]:
                    response_parts.append(f"• {def_sentence}")
                response_parts.append("")
            
            if key_points:
                response_parts.append("**Key Information:**")
                for point in key_points[:3]:
                    response_parts.append(f"• {point}")
                response_parts.append("")
            
            if processes:
                response_parts.append("**Methods & Processes:**")
                for process in processes[:2]:
                    response_parts.append(f"• {process}")
                response_parts.append("")
            
            if examples:
                response_parts.append("**Examples & Applications:**")
                for example in examples[:2]:
                    response_parts.append(f"• {example}")
                response_parts.append("")
        
        # Add context about sources
        if len(chunks) > 1:
            response_parts.append(f"*This information is compiled from {len(chunks)} different sections of your uploaded documents.*")
        
        if all_images:
            response_parts.append(f"*{len(all_images)} related images are available for reference.*")
        
        # Create structured response
        conversational_text = "\n".join(response_parts)
        
        # Calculate relevance score based on match quality
        avg_score = sum([s[1] for s in all_sentences[:3]]) / min(3, len(all_sentences)) if all_sentences else 0
        normalized_score = min(avg_score / 8.0, 1.0)  # Normalize based on new scoring system
        
        return {
            'response': conversational_text,
            'images': all_images,
            'image_contexts': image_contexts,
            'source_chunks': len(chunks),
            'total_images': len(all_images),
            'style': 'conversational',
            'relevance_score': normalized_score
        }
        
    except Exception as e:
        print(f"Error generating ChatGPT-style response: {e}")
        return None

def enhanced_text_search(query, top_k=5, summarize=True):
    """Enhanced search with better relevance scoring"""
    if not document_store['texts']:
        return []
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    results = []
    
    print(f"🔍 Searching for: '{query}' in {len(document_store['texts'])} chunks")
    
    for i, chunk_data in enumerate(document_store['texts']):
        # Handle both old format (string) and new format (dict)
        if isinstance(chunk_data, str):
            text = chunk_data
            images = []
        else:
            text = chunk_data.get('text', '')
            images = chunk_data.get('images', [])
        
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Enhanced scoring system
        score = 0
        
        # 1. Exact phrase matching (highest weight)
        if query_lower in text_lower:
            score += 2.0
        
        # 2. Word overlap scoring
        common_words = query_words.intersection(text_words)
        if query_words:
            word_overlap_score = len(common_words) / len(query_words)
            score += word_overlap_score * 1.5
        
        # 3. Semantic proximity - check for related terms
        semantic_keywords = {
            'ai': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'algorithm'],
            'machine learning': ['ml', 'ai', 'artificial intelligence', 'model', 'training', 'prediction'],
            'neural network': ['nn', 'deep learning', 'ai', 'neuron', 'layer', 'activation'],
            'algorithm': ['method', 'approach', 'technique', 'procedure', 'process'],
            'data': ['information', 'dataset', 'database', 'records', 'statistics'],
            'model': ['algorithm', 'system', 'framework', 'architecture', 'structure'],
            'training': ['learning', 'education', 'teaching', 'instruction', 'practice'],
            'prediction': ['forecast', 'estimate', 'projection', 'anticipation', 'outcome']
        }
        
        for query_word in query_words:
            if query_word in semantic_keywords:
                related_terms = semantic_keywords[query_word]
                for term in related_terms:
                    if term in text_lower:
                        score += 0.3
        
        # 4. Context relevance - boost if query words appear near each other
        words_in_text = [word for word in query_words if word in text_lower]
        if len(words_in_text) > 1:
            # Find positions of query words in text
            word_positions = {}
            for word in words_in_text:
                positions = []
                start = 0
                while True:
                    pos = text_lower.find(word, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                word_positions[word] = positions
            
            # Check if words appear close to each other (within 100 characters)
            for i, word1 in enumerate(words_in_text):
                for word2 in words_in_text[i+1:]:
                    for pos1 in word_positions[word1]:
                        for pos2 in word_positions[word2]:
                            if abs(pos1 - pos2) < 100:
                                score += 0.5
        
        # 5. Length penalty - prefer more substantial content
        if len(text) > 100:
            score += 0.2
        
        print(f"Chunk {i}: score={score:.2f}")
        print(f"Text preview: {text[:150]}...")
        print(f"Images found: {len(images)}")
        
        if score > 0.1:  # Lower threshold for better recall
            results.append({
                'text': text,
                'images': images,
                'score': score,
                'metadata': document_store['metadata'][i] if i < len(document_store['metadata']) else {},
                'rank': len(results) + 1
            })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(results)} matching results")
    
    # If summarize is True, combine top results into single ChatGPT-style response
    if summarize and results:
        top_results = results[:top_k]
        chatgpt_response = generate_chatgpt_style_response(query, top_results)
        
        if chatgpt_response:
            return [{
                'id': str(uuid.uuid4()),
                'title': f"AI Assistant Response",
                'url': '#',
                'snippet': chatgpt_response['response'],  # Send full response instead of truncated
                'full_text': chatgpt_response['response'],
                'images': chatgpt_response['images'],
                'image_contexts': chatgpt_response.get('image_contexts', []),
                'domain': 'AstraFind AI',
                'modality': 'ai_response',
                'score': chatgpt_response.get('relevance_score', 0.8),  # Use normalized score
                'metadata': {
                    'source_chunks': chatgpt_response['source_chunks'],
                    'total_images': chatgpt_response['total_images'],
                    'type': 'ai_response',
                    'style': 'conversational'
                }
            }]
    
    # Return individual results if not summarizing
    return results[:top_k]

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AstraFind Backend",
        "version": "1.0.0",
        "documents_loaded": len(document_store['texts']),
        "processed_files": list(document_store['processed_files'].keys()),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check document store"""
    return jsonify({
        "document_count": len(document_store['texts']),
        "metadata_count": len(document_store['metadata']),
        "processed_files": list(document_store['processed_files'].keys()),
        "sample_text": document_store['texts'][0]['text'][:200] if document_store['texts'] else "No texts",
        "sample_images": len(document_store['texts'][0].get('images', [])) if document_store['texts'] else 0
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Multi-modal search endpoint with summarization"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        summarize = data.get('summarize', True)  # Default to summarization
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        print(f"🔍 Search request: '{query}' (summarize: {summarize})")
        print(f"📚 Document store has {len(document_store['texts'])} chunks")
        
        # Check if we have any documents
        if not document_store['texts']:
            return jsonify({
                "results": [],
                "total": 0,
                "query": query,
                "summarized": summarize,
                "message": "No documents uploaded yet. Please upload a PDF first.",
                "timestamp": datetime.now().isoformat()
            })
        
        # Debug: Print first few chunks
        for i, chunk in enumerate(document_store['texts'][:2]):
            if isinstance(chunk, dict):
                print(f"Chunk {i}: {chunk.get('text', '')[:100]}...")
            else:
                print(f"Chunk {i}: {str(chunk)[:100]}...")
        
        # Perform search with optional summarization
        search_results = enhanced_text_search(query, top_k=5, summarize=summarize)
        print(f"🔍 Search returned {len(search_results)} results")
        
        # Debug: Print search results
        for i, result in enumerate(search_results):
            print(f"Result {i}: type={result.get('metadata', {}).get('type', 'unknown')}, modality={result.get('modality', 'unknown')}")
        
        # Format results for frontend
        formatted_results = []
        for i, result in enumerate(search_results):
            # Check if this is an AI response
            if result.get('metadata', {}).get('type') == 'ai_response':
                formatted_result = {
                    'id': result.get('id', str(uuid.uuid4())),
                    'title': result.get('title', f"AI Assistant Response"),
                    'url': result.get('url', '#'),
                    'snippet': result.get('snippet', ''),
                    'full_text': result.get('full_text', ''),
                    'images': result.get('images', []),
                    'image_contexts': result.get('image_contexts', []),
                    'domain': result.get('domain', 'AstraFind AI'),
                    'modality': result.get('modality', 'ai_response'),
                    'score': result.get('score', 0),
                    'metadata': result.get('metadata', {}),
                    'highlights': [f"AI response from {result.get('metadata', {}).get('source_chunks', 0)} document sections"],
                    'sourceDomain': 'AI Assistant',
                    'publishedAt': datetime.now().isoformat()
                }
            else:
                # Individual chunk result
                formatted_result = {
                    'id': str(uuid.uuid4()),
                    'title': f"Document Match {i+1}",
                    'url': '#',
                    'snippet': result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                    'full_text': result['text'],
                    'images': result.get('images', []),
                    'domain': 'PDF Document',
                    'modality': 'pdf',
                    'score': result.get('score', 0),
                    'metadata': result.get('metadata', {})
                }
            formatted_results.append(formatted_result)
        
        return jsonify({
            "results": formatted_results,
            "total": len(formatted_results),
            "query": query,
            "summarized": summarize,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": "Search failed",
            "message": str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """File upload endpoint with RAG processing"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided",
                "message": "Please select a file to upload"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select a valid file"
            }), 400
        
        # Check file type
        allowed_extensions = {'pdf'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({
                "error": "Invalid file type",
                "message": "Only PDF files are supported for RAG processing"
            }), 400
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save file with unique name
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)
        
        # Process PDF for RAG with rich content
        print(f"Processing PDF: {file.filename}")
        
        # Extract text and images from PDF
        extracted_content = extract_content_from_pdf(file_path)
        if not extracted_content['text'].strip():
            return jsonify({
                "error": "Text extraction failed",
                "message": "Could not extract text from PDF"
            }), 400
        
        # Chunk the content with images
        chunks = chunk_content(extracted_content)
        print(f"Created {len(chunks)} content chunks with images")
        print(f"Total images extracted: {len(extracted_content['images'])}")
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'file_id': file_id,
                'filename': file.filename,
                'file_path': file_path,
                'chunk_id': i,
                'page': i // 3 + 1,  # Approximate page number
                'has_images': len(chunk.get('images', [])) > 0,
                'image_count': len(chunk.get('images', []))
            }
            chunk_metadata.append(metadata)
        
        # Add to document store (store full chunk objects with text and images)
        document_store['texts'].extend(chunks)
        document_store['metadata'].extend(chunk_metadata)
        document_store['processed_files'][file_id] = {
            'filename': file.filename,
            'chunks': len(chunks),
            'processed_at': datetime.now().isoformat()
        }
        
        print(f"RAG processing complete. Total chunks: {len(document_store['texts'])}")
        
        return jsonify({
            "file_id": file_id,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "status": "processed",
            "message": "PDF uploaded and processed for RAG successfully",
            "chunks_created": len(chunks),
            "total_chunks_indexed": len(document_store['texts']),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({
            "error": "Upload failed",
            "message": str(e)
        }), 500

@app.route('/api/filters', methods=['GET'])
def get_filters():
    """Get available filter options"""
    return jsonify({
        "sources": ["web", "pdfs", "images", "tables"],
        "date_ranges": ["any", "day", "week", "month", "year"],
        "file_types": ["pdf", "doc", "docx", "txt", "html"],
        "languages": ["en", "es", "fr", "de", "zh"],
        "sort_options": ["relevance", "date", "popularity"]
    })

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Get search suggestions"""
    query = request.args.get('q', '').lower()
    
    suggestions = [
        "machine learning algorithms",
        "natural language processing",
        "computer vision techniques",
        "deep learning frameworks",
        "AI ethics and bias",
        "transformer architecture",
        "neural network optimization",
        "reinforcement learning",
        "generative AI models",
        "multi-modal AI systems"
    ]
    
    # Filter suggestions based on query
    if query:
        filtered_suggestions = [s for s in suggestions if query in s.lower()]
        return jsonify({"suggestions": filtered_suggestions[:5]})
    
    return jsonify({"suggestions": suggestions[:5]})

@app.route('/api/image/<filename>', methods=['GET'])
def serve_image(filename):
    """Serve uploaded images"""
    try:
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads', 'images')
        file_path = os.path.join(uploads_dir, filename)
        
        if os.path.exists(file_path):
            from flask import send_file
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/url', methods=['POST'])
def load_url():
    """Load content from URL for RAG operations"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({"error": "Invalid URL"}), 400
        
        # Load content from URL
        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to load URL"}), 500
        
        # Extract text and images from HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks_text = '\n'.join(chunk for chunk in lines if chunk)
        
        print(f"Extracted {len(chunks_text)} characters from URL: {url}")
        
        images = []
        
        # Find all images on the page
        for img in soup.find_all('img'):
            img_url = img.get('src')
            if img_url:
                try:
                    # Make absolute URL
                    img_url = urljoin(url, img_url)
                    # Download image (with timeout)
                    img_response = requests.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        # Save image to uploads directory
                        img_filename = f"{uuid.uuid4()}.png"
                        img_path = os.path.join(os.path.dirname(__file__), 'uploads', 'images', img_filename)
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        with open(img_path, 'wb') as img_file:
                            img_file.write(img_response.content)
                        images.append({
                            'page': 1,
                            'index': len(images),
                            'data': f"data:image/png;base64,{base64.b64encode(img_response.content).decode()}",
                            'url': f"/api/image/{img_filename}",
                            'filename': img_filename,
                            'width': 0,
                            'height': 0
                        })
                except Exception as e:
                    print(f"Failed to download image {img_url}: {e}")
        
        # Create proper content structure for chunking
        content_structure = {
            'text': chunks_text,
            'images': images,
            'pages': [{
                'page_num': 1,
                'text': chunks_text,
                'images': images
            }]
        }
        
        # Create chunks from text
        chunks = chunk_content(content_structure)
        print(f"Created {len(chunks)} chunks from URL content")
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'url': url,
                'source': 'web',
                'chunk_id': i,
                'page': 1,
                'has_images': len(chunk.get('images', [])) > 0,
                'image_count': len(chunk.get('images', []))
            }
            chunk_metadata.append(metadata)
        
        # Add to document store
        document_store['texts'].extend(chunks)
        document_store['metadata'].extend(chunk_metadata)
        
        print(f"Added {len(chunks)} chunks to document store. Total: {len(document_store['texts'])}")
        
        return jsonify({
            "url": url,
            "chunks_created": len(chunks),
            "total_chunks_indexed": len(document_store['texts']),
            "text_length": len(chunks_text),
            "images_found": len(images),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to load URL",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting AstraFind Backend Server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🔍 API endpoints:")
    print("   - GET  /                 - Health check")
    print("   - POST /api/search       - Multi-modal search")
    print("   - POST /api/upload       - File upload")
    print("   - GET  /api/filters      - Available filters")
    print("   - GET  /api/suggestions  - Search suggestions")
    print("   - GET  /api/image/<filename> - Serve images")
    print("   - POST /api/url          - Load content from URL")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
