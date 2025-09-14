# 📚 RAG Document Q&A System

A powerful Retrieval-Augmented Generation (RAG) system that allows you to upload documents and ask questions about them using Google Gemini AI. The system uses advanced semantic search to find relevant content and generates comprehensive answers based on the document context.

## 🚀 Features

- **Multi-format Document Support**: Upload PDF, DOCX, and TXT files
- **Intelligent Document Processing**: Automatic text extraction and chunking
- **Semantic Search**: Uses sentence transformers for finding relevant content
- **Google Gemini Integration**: Powered by Google's Gemini 1.5 Flash model (free tier)
- **Interactive Chat Interface**: Streamlit-based user-friendly interface
- **Source Attribution**: Shows relevant document chunks used for answers
- **Real-time Processing**: Process documents and get answers instantly
- **Chat Memory Saving and document caching**: includes a built-in memory to store previous chat history to answer questions accordingly and also contains document caching which reduces upload time.
## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF2, python-docx
- **Environment Management**: python-dotenv

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key (free tier available)
- Virtual environment (recommended)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag-chat
```

### 2. Create Virtual Environment
```bash
python -m venv rag_env
source rag_env/bin/activate  # On Linux/Mac
# rag_env\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🔑 Getting Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click on "Get API key"
4. Create a new API key
5. Copy the key to your `.env` file

**Note**: The free tier of Gemini API has generous limits and is perfect for personal projects.

## 🚀 Usage

### Running the Application
```bash
source rag_env/bin/activate  # Activate virtual environment
streamlit run rag_app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **API Key Setup**: The application automatically loads your Gemini API key from the `.env` file
2. **Upload Document**: Use the sidebar to upload a PDF, DOCX, or TXT file
3. **Process Document**: Click "Process Document" to analyze and chunk your document
4. **Ask Questions**: Type your questions in the chat interface
5. **View Sources**: Check the right sidebar to see relevant document chunks used for answers

## 📁 Project Structure

```
rag-chat/
├── rag_app.py              # Main application file
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (API key)
├── .gitignore            # Git ignore file
├── README.md             # This file
└── rag_env/              # Virtual environment (created after setup)
```

## 🏗️ Architecture

### Document Processing Pipeline
1. **Text Extraction**: Extract text from uploaded files
2. **Chunking**: Split documents into overlapping chunks (500 chars, 50 overlap)
3. **Embedding**: Generate vector embeddings using SentenceTransformers
4. **Indexing**: Store embeddings in FAISS vector database

### Query Processing Pipeline
1. **Query Embedding**: Convert user question to vector embedding
2. **Similarity Search**: Find top-k relevant document chunks
3. **Context Building**: Combine relevant chunks as context
4. **Answer Generation**: Use Gemini to generate answer based on context

## ⚙️ Configuration

### Customizable Parameters
You can modify these constants in `rag_app.py`:

```python
CHUNK_SIZE = 500        # Size of text chunks
CHUNK_OVERLAP = 50      # Overlap between chunks
TOP_K_CHUNKS = 3        # Number of chunks to retrieve
```

### Model Configuration
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM Model**: `gemini-1.5-flash` (Google's free tier model)

## 🔍 Supported File Formats

- **PDF**: Text-based PDFs (not scanned images)
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files

## 💡 Example Use Cases

- **Research Papers**: Ask questions about academic papers
- **Technical Documentation**: Query manuals and guides
- **Legal Documents**: Analyze contracts and agreements
- **Educational Content**: Study materials and textbooks
- **Business Reports**: Extract insights from reports

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: No Gemini API key found
   ```
   **Solution**: Ensure your `.env` file contains `GEMINI_API_KEY=your_key_here`

2. **Model Not Found Error**
   ```
   Error: models/gemini-pro is not found
   ```
   **Solution**: The app uses `gemini-1.5-flash` for free tier compatibility

3. **Document Processing Failed**
   **Solution**: Ensure the document is not password-protected and is a valid format

4. **Memory Issues with Large Documents**
   **Solution**: Reduce `CHUNK_SIZE` or process smaller documents

### Performance Tips

- **Optimal Document Size**: 1-50 pages work best
- **Clear Questions**: Specific questions yield better answers
- **Document Quality**: Well-formatted documents produce better results

## 🔒 Security & Privacy

- **Local Processing**: Document processing happens locally
- **API Calls**: Only processed chunks are sent to Gemini API
- **No Data Storage**: Documents are not permanently stored
- **Environment Variables**: API keys are stored securely in `.env`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini**: For providing the powerful LLM API
- **Sentence Transformers**: For excellent embedding models
- **FAISS**: For efficient vector similarity search
- **Streamlit**: For the amazing web framework
- **Hugging Face**: For the transformer models

## 📊 System Requirements

- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB for model downloads
- **Internet**: Required for initial model download and API calls

## 📞 Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Happy Document Querying! 🎉**
# chat_with_pdf_rag
