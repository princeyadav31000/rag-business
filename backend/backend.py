from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from dataclasses import dataclass
import pickle
import google.generativeai as genai
import os
import PyPDF2
import io
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class EnhancedAirtableRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", gemini_api_key: Optional[str] = None):
        self.embedding_model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25 = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.tfidf_matrix = None
        self.processed_documents = []
        self.chunk_embeddings = None
        self.chunks = []

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            self.stop_words = set(stopwords.words('english'))
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key must be provided either as parameter or GEMINI_API_KEY environment variable")
            genai.configure(api_key=api_key)
        
        self.gemini_model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')
        
    def preprocess_record(self, record: Dict[str, Any]) -> str:
        text_parts = []
        
        name = record.get("Name", "Unknown")
        text_parts.append(f"Name: {name}")
        
        if "One Sentence Description" in record and record["One Sentence Description"]:
            text_parts.append(f"Description: {record['One Sentence Description']}")
        
        if "Area" in record and record["Area"]:
            areas = record["Area"] if isinstance(record["Area"], list) else [record["Area"]]
            text_parts.append(f"Areas: {', '.join(areas)}")
        
        if "Type" in record:
            text_parts.append(f"Type: {record['Type']}")
        
        if "Link" in record:
            text_parts.append(f"Link: {record['Link']}")
        
        if "Notes" in record and record["Notes"]:
            text_parts.append(f"Notes: {record['Notes']}")
        
        if "documentText" in record and record["documentText"]:
            # Clean document text but keep more content
            doc_text = self.clean_text(record["documentText"])
            text_parts.append(f"Document Content: {doc_text}")
        
        if "youtubeTranscript" in record and record["youtubeTranscript"]:
            transcript = self.clean_text(record["youtubeTranscript"])
            text_parts.append(f"YouTube Transcript: {transcript}")
        
        if "Attachments" in record and record["Attachments"]:
            attachment_names = []
            for att in record["Attachments"]:
                if "filename" in att:
                    attachment_names.append(att["filename"])
            if attachment_names:
                text_parts.append(f"Attachments: {', '.join(attachment_names)}")
        
        return " | ".join(text_parts)
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\-()"\']', '', text)
        return text.strip()

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        processed_query = {
            'original': query,
            'cleaned': query.lower().strip(),
            'entities': [],
            'keywords': [],
            'intent': 'search',
            'expanded_terms': []
        }
        
        tokens = word_tokenize(query.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]
        processed_query['keywords'] = tokens
        
        if self.nlp:
            doc = self.nlp(query)
            processed_query['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
        
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(word in query.lower() for word in question_words):
            processed_query['intent'] = 'question'
        elif any(word in query.lower() for word in ['find', 'search', 'look', 'show']):
            processed_query['intent'] = 'search'
        
        expanded_terms = []
        for token in tokens:
            if len(token) > 3:
                expanded_terms.append(token)
                if token.endswith('s'):
                    expanded_terms.append(token[:-1])
        
        processed_query['expanded_terms'] = list(set(expanded_terms))
        processed_query['expanded_query'] = ' '.join(processed_query['expanded_terms'])
        
        return processed_query

    def smart_chunk_document(self, text: str, doc_metadata: Dict) -> List[Dict]:
        chunks = []
        
        doc_type = doc_metadata.get('Type', '').lower()
        
        if 'youtube' in doc_type or 'youtubeTranscript' in doc_metadata:
            sentences = re.split(r'[.!?]+', text)
            chunk_size = 5
            overlap = 2
            
            for i in range(0, len(sentences), chunk_size - overlap):
                chunk_sentences = sentences[i:i + chunk_size]
                chunk_text = '. '.join(chunk_sentences).strip()
                if len(chunk_text) > 50:
                    chunks.append({
                        'text': chunk_text,
                        'type': 'transcript_chunk',
                        'position': i,
                        'metadata': doc_metadata
                    })
        
        elif 'document' in doc_type.lower() or 'documentText' in doc_metadata:
            paragraphs = re.split(r'\n\s*\n', text)
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 100:
                    if len(paragraph) > 1000:
                        sub_chunks = self.split_into_chunks(paragraph, chunk_size=800, overlap=100)
                        for j, sub_chunk in enumerate(sub_chunks):
                            chunks.append({
                                'text': sub_chunk,
                                'type': 'document_chunk',
                                'position': f"{i}.{j}",
                                'metadata': doc_metadata
                            })
                    else:
                        chunks.append({
                            'text': paragraph.strip(),
                            'type': 'document_chunk',
                            'position': i,
                            'metadata': doc_metadata
                        })
        
        else:
            default_chunks = self.split_into_chunks(text, chunk_size=600, overlap=100)
            for i, chunk in enumerate(default_chunks):
                chunks.append({
                    'text': chunk,
                    'type': 'default_chunk',
                    'position': i,
                    'metadata': doc_metadata
                })
        
        return chunks
    
    def load_data(self, json_file_path: str):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            records = [data]
        else:
            records = data
        
        print(f"Loading {len(records)} records...")
        
        for i, record in enumerate(records):
            content = self.preprocess_record(record)
            doc = Document(
                id=str(i),
                content=content,
                metadata=record
            )
            self.documents.append(doc)
        
        print(f"Successfully loaded {len(self.documents)} documents")
    
    def create_embeddings(self):
        if not self.documents:
            raise ValueError("No documents loaded. Please load data first.")
        
        print("Creating embeddings...")

        contents = []
        processed_docs = []
        all_chunks = []

        for doc in self.documents:
            contents.append(doc.content)
            processed_docs.append(doc.content.lower())
            
            chunks = self.smart_chunk_document(doc.content, doc.metadata)
            for chunk in chunks:
                all_chunks.append(chunk)
                contents.append(chunk['text'])
        
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)

        doc_count = len(self.documents)
        for i, doc in enumerate(self.documents):
            doc.embedding = embeddings[i]
        
        self.chunk_embeddings = embeddings[doc_count:]
        self.chunks = all_chunks
        
        self.embeddings = np.array(embeddings[:doc_count])

        print("Preparing BM25 index...")
        tokenized_docs = [word_tokenize(doc.lower()) for doc in processed_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print("Preparing TF-IDF matrix...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_docs)
        self.processed_documents = processed_docs
        
        print(f"Created embeddings for {len(self.documents)} documents and {len(all_chunks)} chunks")

    def semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                'score': float(similarities[idx]),
                'content': doc.content,
                'metadata': doc.metadata,
                'id': doc.id,
                'search_type': 'semantic'
            })
        
        return results
    
    def bm25_search(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'score': float(scores[idx]),
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'id': doc.id,
                    'search_type': 'bm25'
                })
        
        return results

    def tfidf_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.tfidf_matrix is None:
            return []
        
        query_vec = self.tfidf_vectorizer.transform([query.lower()])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'score': float(scores[idx]),
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'id': doc.id,
                    'search_type': 'tfidf'
                })
        
        return results

    def combine_search_results(self, semantic_results: List, bm25_results: List, 
                             tfidf_results: List, processed_query: Dict, top_k: int) -> List[Dict[str, Any]]:
        
        combined_scores = {}
        
        weights = {
            'semantic': 0.5,
            'bm25': 0.3,
            'tfidf': 0.2
        }
        
        all_results = semantic_results + bm25_results + tfidf_results
        
        for result in all_results:
            doc_id = result['id']
            search_type = result['search_type']
            score = result['score']
            
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'document': result,
                    'scores': {'semantic': 0, 'bm25': 0, 'tfidf': 0},
                    'combined_score': 0
                }
            
            combined_scores[doc_id]['scores'][search_type] = max(
                combined_scores[doc_id]['scores'][search_type], score
            )
        
        for doc_id, data in combined_scores.items():
            scores = data['scores']
            combined_score = sum(scores[method] * weights[method] for method in weights)
            
            if processed_query['entities']:
                entity_boost = 0
                doc_content = data['document']['content'].lower()
                for entity, _ in processed_query['entities']:
                    if entity.lower() in doc_content:
                        entity_boost += 0.1
                combined_score += entity_boost
            
            data['combined_score'] = combined_score
        
        sorted_results = sorted(combined_scores.values(), 
                              key=lambda x: x['combined_score'], reverse=True)
        
        final_results = []
        for item in sorted_results[:top_k]:
            result = item['document'].copy()
            result['score'] = item['combined_score']
            result['method_scores'] = item['scores']
            final_results.append(result)
        
        return final_results
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        processed_query = self.preprocess_query(query)
        
        semantic_results = self.semantic_search(processed_query['expanded_query'], top_k)
        bm25_results = self.bm25_search(processed_query['keywords'], top_k)
        tfidf_results = self.tfidf_search(query, top_k)
        
        combined_results = self.combine_search_results(
            semantic_results, bm25_results, tfidf_results, 
            processed_query, top_k
        )
        
        return combined_results
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Please run create_embeddings() first.")
        
        return self.hybrid_search(query, top_k)
    
    def split_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def find_relevant_chunks(self, chunks: List[str], query: str, top_k: int = 5) -> List[str]:
        if not chunks:
            return []
        
        chunk_embeddings = self.embedding_model.encode(chunks)
        query_embedding = self.embedding_model.encode([query])
        
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [chunks[i] for i in top_indices]
    
    def prepare_context_for_gemini(self, top_result: Dict[str, Any], question:str) -> str:
        metadata = top_result['metadata']
        context_parts = []
        
        name = metadata.get('Name', 'Unknown')
        context_parts.append(f"**Resource Name:** {name}")
        
        if 'One Sentence Description' in metadata and metadata['One Sentence Description']:
            context_parts.append(f"**Description:** {metadata['One Sentence Description']}")
        
        if 'Type' in metadata and metadata['Type']:
            context_parts.append(f"**Type:** {metadata['Type']}")
        
        if 'Area' in metadata and metadata['Area']:
            areas = metadata['Area'] if isinstance(metadata['Area'], list) else [metadata['Area']]
            context_parts.append(f"**Areas:** {', '.join(areas)}")
        
        if 'Link' in metadata and metadata['Link']:
            context_parts.append(f"**Link:** {metadata['Link']}")
        
        if 'Notes' in metadata and metadata['Notes']:
            context_parts.append(f"**Notes:** {metadata['Notes']}")
        
        if 'documentText' in metadata and metadata['documentText']:
            doc_text = self.clean_text(metadata['documentText'])
            doc_chunks = self.split_into_chunks(doc_text, chunk_size=1000, overlap=100)
            doc_relevant_chunks = self.find_relevant_chunks(doc_chunks, question, top_k=5)
            doc_text = " ... ".join(doc_relevant_chunks)
            context_parts.append(f"**Document Content:** {doc_text}")
        
        if 'youtubeTranscript' in metadata and metadata['youtubeTranscript']:
            transcript = self.clean_text(metadata['youtubeTranscript'])
            youtube_chunks = self.split_into_chunks(transcript, chunk_size=1000, overlap=100)
            youtube_relevant_chunks = self.find_relevant_chunks(youtube_chunks, question, top_k=5)
            transcript = " ... ".join(youtube_relevant_chunks)
            context_parts.append(f"**YouTube Transcript:** {transcript}")
        
        if 'Attachments' in metadata and metadata['Attachments']:
            attachment_info = []
            for att in metadata['Attachments']:
                filename = att.get('filename', 'Unknown file')
                file_type = att.get('type', 'Unknown type')
                attachment_info.append(f"{filename} ({file_type})")
            context_parts.append(f"**Attachments:** {', '.join(attachment_info)}")
        
        return "\n\n".join(context_parts)
    
    def answer_question_with_gemini(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        try:
            results = self.search(question, top_k=top_k)
            
            if not results:
                return self._answer_with_gemini_knowledge(question)
            
            top_result = results[0]
            
            if top_result['score'] < 0.1:
                return self._answer_with_gemini_knowledge(question)
            
            context = self.prepare_context_for_gemini(top_result, question)
            
            prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context. You must not use any external knowledge or information outside of what's given in the context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context above
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
3. Be specific and detailed in your response when the context allows
4. If there are links, URLs, or attachments mentioned in the context, include them in your response
5. Structure your answer clearly and professionally
6. Do not make up or infer information that isn't explicitly stated in the context

ANSWER:"""

            response = self.gemini_model.generate_content(prompt)
            
            if "don't have enough information" in response.text.lower() or "insufficient information" in response.text.lower():
                return self._answer_with_gemini_knowledge(question)

            return {
                "answer": response.text,
                "source": top_result['metadata'].get('Name', 'Unknown'),
                "score": top_result['score'],
                "link": top_result['metadata'].get('Link', None),
                "type": top_result['metadata'].get('Type', None),
                "areas": top_result['metadata'].get('Area', []),
                "method_scores": top_result.get('method_scores', {})
            }
            
        except Exception as e:
            print(f"Error generating response with Gemini: {e}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "source": None,
                "score": 0.0
            }
    
    def _answer_with_gemini_knowledge(self, question: str) -> Dict[str, Any]:
        """Use Gemini's own knowledge when no relevant context is found"""
        try:
            prompt = f"""You are a helpful assistant. Answer the following question using your knowledge. Be informative and helpful.

    QUESTION: {question}

    Please provide a clear and comprehensive answer."""

            response = self.gemini_model.generate_content(prompt)
            
            return {
                "answer": response.text + "\n\n*Note: This answer is based on general knowledge as no specific information was found in the database.*",
                "source": "General Knowledge",
                "score": 0.0,
                "link": None,
                "type": "General Knowledge",
                "areas": []
            }
            
        except Exception as e:
            print(f"Error generating response with Gemini knowledge: {e}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "source": None,
                "score": 0.0
            }

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.documents = model_data['documents']
        self.embeddings = model_data['embeddings']
        self.chunk_embeddings = model_data.get('chunk_embeddings')
        self.chunks = model_data.get('chunks', [])
        self.processed_documents = model_data.get('processed_documents', [])
        self.tfidf_vectorizer = model_data.get('tfidf_vectorizer')
        self.tfidf_matrix = model_data.get('tfidf_matrix')
        self.bm25 = model_data.get('bm25')
        print(f"Model loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {"message": "No documents loaded"}
        
        stats = {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks) if self.chunks else 0,
            "documents_with_links": sum(1 for doc in self.documents if doc.metadata.get('Link')),
            "documents_with_attachments": sum(1 for doc in self.documents if doc.metadata.get('Attachments')),
            "documents_with_notes": sum(1 for doc in self.documents if doc.metadata.get('Notes')),
            "documents_with_document_text": sum(1 for doc in self.documents if doc.metadata.get('documentText')),
            "documents_with_youtube_transcript": sum(1 for doc in self.documents if doc.metadata.get('youtubeTranscript')),
            "unique_types": list(set(doc.metadata.get('Type', 'Unknown') for doc in self.documents)),
            "unique_areas": []
        }
        
        all_areas = []
        for doc in self.documents:
            areas = doc.metadata.get('Area', [])
            if areas:
                if isinstance(areas, list):
                    all_areas.extend(areas)
                else:
                    all_areas.append(areas)
        
        stats["unique_areas"] = list(set(all_areas))
        
        return stats

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

rag_instance = None
pdf_documents = []
pdf_embeddings = None

def initialize_rag():
    global rag_instance
    try:
        rag_instance = EnhancedAirtableRAG(gemini_api_key="AIzaSyD6XF8GmoHEn4BPnvPrjRtYN7XcMZHB4-o")
        
        model_file = "rag_model.pkl"
        try:
            rag_instance.load_model(model_file)
            print("✅ RAG model loaded successfully!")
            return True
        except FileNotFoundError:
            print("❌ No existing model found. Please ensure rag_model.pkl exists.")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing RAG: {e}")
        return False

def process_pdf(file_content):
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def create_pdf_embeddings(pdf_text, filename):
    """Create embeddings for PDF content"""
    global pdf_documents, pdf_embeddings
    
    if not rag_instance:
        return False
    
    # Split PDF into chunks
    chunks = rag_instance.split_into_chunks(pdf_text, chunk_size=1000, overlap=100)
    
    pdf_documents = []
    for i, chunk in enumerate(chunks):
        pdf_documents.append({
            'id': f"pdf_{i}",
            'content': chunk,
            'filename': filename,
            'chunk_index': i
        })
    
    # Create embeddings
    contents = [doc['content'] for doc in pdf_documents]
    pdf_embeddings = rag_instance.embedding_model.encode(contents)
    
    return True

def search_pdf(query, top_k=3):
    """Search in uploaded PDF"""
    global pdf_documents, pdf_embeddings
    
    if not pdf_documents or pdf_embeddings is None:
        return []
    
    query_embedding = rag_instance.embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, pdf_embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum threshold
            results.append({
                'content': pdf_documents[idx]['content'],
                'filename': pdf_documents[idx]['filename'],
                'chunk_index': pdf_documents[idx]['chunk_index'],
                'score': float(similarities[idx])
            })
    
    return results

def answer_with_pdf_context(question, pdf_results):
    """Generate answer using PDF context"""
    if not pdf_results:
        return None
    
    context = "\n\n".join([result['content'] for result in pdf_results])
    
    prompt = f"""Based on the following document content, answer the question. If the content doesn't contain enough information, say so.

DOCUMENT CONTENT:
{context}

QUESTION: {question}

ANSWER:"""

    try:
        response = rag_instance.gemini_model.generate_content(prompt)
        return {
            "answer": response.text,
            "source": f"PDF: {pdf_results[0]['filename']}",
            "score": pdf_results[0]['score'],
            "source_type": "pdf"
        }
    except Exception as e:
        print(f"Error with Gemini: {e}")
        return None

def combine_pdf_rag_context(pdf_results, rag_result, question):
    """Combine PDF and RAG context for Gemini answer"""
    pdf_context = "\n\n".join([result['content'] for result in pdf_results])
    rag_context = rag_instance.prepare_context_for_gemini(rag_result, question)
    combined_context = f"PDF Content:\n{pdf_context}\n\nRAG Content:\n{rag_context}"
    return combined_context

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "RAG API is running",
        "model_loaded": rag_instance is not None
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    if not rag_instance:
        return jsonify({"error": "RAG model not initialized"}), 500
    
    try:
        stats = rag_instance.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if not rag_instance:
        return jsonify({"error": "RAG model not initialized"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    try:
        file_content = file.read()
        pdf_text = process_pdf(file_content)
        
        if not pdf_text:
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        success = create_pdf_embeddings(pdf_text, file.filename)
        
        if success:
            return jsonify({
                "message": "PDF uploaded and processed successfully",
                "filename": file.filename,
                "chunks_created": len(pdf_documents)
            })
        else:
            return jsonify({"error": "Failed to process PDF"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

@app.route('/clear-pdf', methods=['POST'])
def clear_pdf():
    global pdf_documents, pdf_embeddings
    pdf_documents = []
    pdf_embeddings = None
    return jsonify({"message": "PDF data cleared"})

@app.route('/chat', methods=['POST'])
def chat():
    if not rag_instance:
        return jsonify({"error": "RAG model not initialized"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400

        query = data['message'].strip()
        top_k = data.get('top_k', 5)
        search_mode = data.get('search_mode', 'both')
        print(search_mode)
        
        if not query:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        pdf_response = None
        pdf_results = []
        if search_mode in ['pdf', 'both'] and pdf_documents:
            pdf_results = search_pdf(query)
            print(f"PDF search results: {len(pdf_results)} found")
            if pdf_results:
                pdf_response = answer_with_pdf_context(query, pdf_results)

        rag_response = None
        rag_result = None
        if search_mode in ['rag', 'both']:
            rag_response = rag_instance.answer_question_with_gemini(query)
            # For context, get the top result (if any)
            results = rag_instance.search(query, top_k=1) if hasattr(rag_instance, 'search') else []
            if results:
                rag_result = results[0]

        final_response = None
        pdf_score = pdf_response['score'] if pdf_response else 0.0
        rag_score = rag_response['score'] if rag_response else 0.0
        pdf_threshold = 0.3
        rag_threshold = 0.1
        
        if search_mode == 'pdf' and pdf_response:
            final_response = pdf_response
        elif search_mode == 'rag' and rag_response:
            final_response = rag_response
        elif search_mode == 'both':
            # Both have good results
            if pdf_response and rag_response and pdf_score > pdf_threshold and rag_score > rag_threshold:
                # Combine both contexts and ask Gemini
                combined_context = combine_pdf_rag_context(pdf_results, rag_result, query)
                prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context. You must not use any external knowledge or information outside of what's given in the context.\n\nCONTEXT:\n{combined_context}\n\nQUESTION: {query}\n\nINSTRUCTIONS:\n1. Answer the question using ONLY the information provided in the context above\n2. If the context doesn't contain enough information to answer the question, say \"I don't have enough information in the provided context to answer this question.\"\n3. Be specific and detailed in your response when the context allows\n4. If there are links, URLs, or attachments mentioned in the context, include them in your response\n5. Structure your answer clearly and professionally\n6. Do not make up or infer information that isn't explicitly stated in the context\n\nANSWER:"""
                try:
                    response = rag_instance.gemini_model.generate_content(prompt)
                    answer = response.text
                    if "don't have enough information" in answer.lower() or "insufficient information" in answer.lower():
                        # fallback to general knowledge
                        final_response = rag_instance._answer_with_gemini_knowledge(query)
                    else:
                        final_response = {
                            "answer": answer,
                            "source": f"PDF: {pdf_results[0]['filename']} + RAG: {rag_result['metadata'].get('Name', 'Unknown') if rag_result else 'Unknown'}",
                            "score": max(pdf_score, rag_score),
                            "link": rag_result['metadata'].get('Link', None) if rag_result else None,
                            "type": rag_result['metadata'].get('Type', None) if rag_result else None,
                            "areas": rag_result['metadata'].get('Area', []) if rag_result else [],
                            "source_type": "pdf+rag"
                        }
                except Exception as e:
                    print(f"Error with Gemini (combined context): {e}")
                    final_response = rag_instance._answer_with_gemini_knowledge(query)
            elif pdf_response and pdf_score > pdf_threshold:
                final_response = pdf_response
            elif rag_response and rag_score > rag_threshold:
                final_response = rag_response
            else:
                final_response = pdf_response or rag_response

        if not final_response:
            # Fallback to general knowledge
            final_response = rag_instance._answer_with_gemini_knowledge(query)
        
        return jsonify({
            "message": final_response["answer"],
            "source": final_response.get("source"),
            "score": final_response.get("score", 0.0),
            "link": final_response.get("link"),
            "type": final_response.get("type"),
            "areas": final_response.get("areas", []),
            "source_type": final_response.get("source_type", "rag"),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting RAG API Server...")
    
    # Initialize RAG model
    if initialize_rag():
        print("🎉 RAG model initialized successfully!")
        print("📊 Available endpoints:")
        print("  - GET  /health - Health check")
        print("  - GET  /stats  - Get data statistics")
        print("  - POST /chat   - Ask questions")
        print("\n🌐 Starting server on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize RAG model. Please check your setup.")
        print("Make sure you have:")
        print("1. rag_model.pkl file in the same directory")
        print("2. Valid Gemini API key")
        print("3. All required Python packages installed")