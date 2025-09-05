import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import io
import re
import hashlib
from dataclasses import dataclass
import pickle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Document processing imports
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# LLM integration (using Google Gemini)
import google.generativeai as genai

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_CHUNKS = 3

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class DocumentProcessor:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []

    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""

    def extract_text_from_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            text = file.read().decode('utf-8')
            return text
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""

    def chunk_text(self, text: str, filename: str) -> List[DocumentChunk]:
        """Split text into chunks with overlap"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()

        chunks = []
        words = text.split()

        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text.strip()) > 0:
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        'filename': filename,
                        'chunk_id': len(chunks),
                        'start_word': i,
                        'end_word': min(i + CHUNK_SIZE, len(words))
                    }
                )
                chunks.append(chunk)

        return chunks

class VectorStore:
    def __init__(self):
        self.embedder = None
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.dimension = 384  # Default for sentence-transformers models

    def initialize_embedder(self):
        """Initialize the sentence transformer model"""
        if self.embedder is None:
            with st.spinner("Loading embedding model..."):
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = self.embedder.get_sentence_embedding_dimension()

    def create_embeddings(self, chunks: List[DocumentChunk]):
        """Create embeddings for chunks and build FAISS index"""
        self.initialize_embedder()

        with st.spinner("Creating embeddings..."):
            # Extract texts
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings
            embeddings = self.embedder.encode(texts, show_progress_bar=True)

            # Store embeddings in chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]

            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add to index
            self.index.add(embeddings.astype('float32'))

            # Store chunks
            self.chunks = chunks

    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[DocumentChunk]:
        """Search for similar chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []

        # Encode query
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Return relevant chunks
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk.metadata['similarity_score'] = float(scores[0][i])
                relevant_chunks.append(chunk)

        return relevant_chunks

class RAGSystem:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.client = None

    def initialize_llm(self, api_key: str = None):
        """Initialize Gemini client"""
        try:
            # Use API key from parameter or environment variable
            if api_key:
                genai.configure(api_key=api_key)
            else:
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    st.error("No Gemini API key found in environment variables or provided")
                    return False
                genai.configure(api_key=api_key)

            # Test the connection by creating a model
            self.client = genai.GenerativeModel('gemini-1.5-flash')
            return True
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return False

    def process_document(self, uploaded_file):
        """Process uploaded document"""
        filename = uploaded_file.name
        file_extension = filename.lower().split('.')[-1]

        # Extract text based on file type
        if file_extension == 'pdf':
            text = self.document_processor.extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            text = self.document_processor.extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            text = self.document_processor.extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file format")
            return False

        if not text.strip():
            st.error("No text extracted from the document")
            return False

        # Chunk the text
        chunks = self.document_processor.chunk_text(text, filename)

        if not chunks:
            st.error("No chunks created from the document")
            return False

        # Create embeddings and vector store
        self.vector_store.create_embeddings(chunks)

        return True

    def generate_answer(self, query: str, relevant_chunks: List[DocumentChunk]) -> str:
        """Generate answer using LLM"""
        if not self.client:
            return "LLM not initialized. Please provide a valid Gemini API key."

        # Create context from relevant chunks
        context = "\n\n".join([
            f"[Chunk {chunk.metadata['chunk_id']}]: {chunk.content}"
            for chunk in relevant_chunks
        ])

        prompt = f"""Based on the following context from the document, please answer the user's question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the information in the context. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""

        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def query(self, question: str) -> Dict[str, Any]:
        """Main query function"""
        # Search for relevant chunks
        relevant_chunks = self.vector_store.search(question)

        if not relevant_chunks:
            return {
                "answer": "I couldn't find any relevant information in the document to answer your question.",
                "chunks": [],
                "sources": []
            }

        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)

        return {
            "answer": answer,
            "chunks": relevant_chunks,
            "sources": [chunk.metadata for chunk in relevant_chunks]
        }

def main():
    st.set_page_config(
        page_title="RAG Document Q&A",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 RAG Document Q&A System")
    st.markdown("Upload a document and ask questions about it using Google Gemini AI!")

    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Initialize LLM with environment variable
        if not st.session_state.rag_system.client:
            if st.session_state.rag_system.initialize_llm():
                st.success("✅ Gemini LLM initialized successfully!")
            else:
                st.error("❌ Failed to initialize Gemini LLM. Please check your API key in .env file")

        # Optional: Allow manual API key input
        with st.expander("🔧 Manual API Key Override"):
            api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key (optional - overrides .env file)")

            if api_key and st.button("Update API Key"):
                if st.session_state.rag_system.initialize_llm(api_key):
                    st.success("✅ Gemini LLM updated successfully!")
                else:
                    st.error("❌ Failed to initialize Gemini LLM")

        st.divider()

        # Document upload
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload a PDF, TXT, or DOCX file"
        )

        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                if st.session_state.rag_system.process_document(uploaded_file):
                    st.session_state.document_processed = True
                    st.session_state.chat_history = []  # Clear chat history
                    st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                    st.rerun()

        # Document info
        if st.session_state.document_processed:
            st.info(f"📊 Document loaded with {len(st.session_state.rag_system.vector_store.chunks)} chunks")

        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])

        # Chat input
        if st.session_state.document_processed and st.session_state.rag_system.client:
            query = st.chat_input("Ask a question about the document...")

            if query:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": query})

                # Process query
                with st.spinner("Searching and generating answer..."):
                    result = st.session_state.rag_system.query(query)

                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})

                # Store sources for display
                if 'last_sources' not in st.session_state:
                    st.session_state.last_sources = []
                st.session_state.last_sources = result["sources"]

                st.rerun()

        elif not st.session_state.document_processed:
            st.info("📤 Please upload and process a document first.")
        elif not st.session_state.rag_system.client:
            st.info("🔑 Please check your Gemini API key configuration.")

    with col2:
        st.header("📎 Sources & Context")

        if hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
            st.subheader("Recent Query Sources:")
            for i, source in enumerate(st.session_state.last_sources, 1):
                with st.expander(f"Source {i} (Score: {source.get('similarity_score', 0):.3f})"):
                    st.write(f"**File:** {source['filename']}")
                    st.write(f"**Chunk ID:** {source['chunk_id']}")

                    # Find the actual chunk content
                    for chunk in st.session_state.rag_system.vector_store.chunks:
                        if chunk.metadata['chunk_id'] == source['chunk_id']:
                            st.write("**Content:**")
                            st.text_area("", chunk.content, height=150, key=f"chunk_{i}")
                            break

if __name__ == "__main__":
    main()
