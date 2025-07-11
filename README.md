# Medical Chatbot with RAG (Retrieval-Augmented Generation)

A Flask-based medical chatbot that uses Retrieval-Augmented Generation (RAG) to answer medical questions based on PDF documents. The chatbot leverages HuggingFace embeddings, Pinecone vector database, and Cohere's language model to provide accurate medical information.

## 🚀 Features

- **RAG Architecture**: Combines document retrieval with language generation for accurate responses
- **Dynamic Prompting**: Adjusts response length based on query complexity
- **Medical Focus**: Specialized for medical questions and information
- **PDF Document Processing**: Ingests and processes medical PDF documents
- **Vector Search**: Uses Pinecone for efficient similarity search
- **Web Interface**: Simple Flask-based web interface for easy interaction

## 📁 Project Structure

```
medical-chatbot/
├── data/                    # PDF documents folder
│   └── *.pdf               # Medical books/documents
├── utils/                  # Utility functions
│   └── ingest.py          # Document loading and processing
├── templates/             # HTML templates
│   └── index.html         # Web interface
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
├── app.py                # Main Flask application
└── README.md             # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Pinecone account and API key
- Cohere API key

### 1. Clone the Repository

```bash
git clone <repository-url>
cd medical-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
COHERE_API_KEY=your_cohere_api_key_here
```

### 4. Prepare Your Data

1. Place your medical PDF documents in the `data/` folder
2. The system will automatically process all PDF files in this directory

### 5. Set Up Pinecone Index

Uncomment the index creation code in `app.py` for first-time setup:

```python
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
```

Also uncomment the document vectorization code:

```python
text_vector = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)
```

Run the application once to create the index and upload documents, then comment these sections back.

## 🚀 Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:8080`

## 📖 Usage

1. **Web Interface**: Navigate to `http://localhost:8080` in your browser
2. **Ask Questions**: Type medical questions in the chat interface
3. **Get Responses**: The chatbot will provide relevant answers based on your PDF documents

### Query Types

- **Short queries** (≤6 words): Get concise 1-2 sentence answers
- **Long queries** (>6 words): Get detailed 5-7 sentence explanations

## 🔧 Key Components

### app.py
- **Main Flask application**
- **RAG chain setup** with retrieval and generation
- **Dynamic prompting** based on query length
- **API endpoints** for web interface

### utils/ingest.py
- **Document loading** from PDF files
- **Text splitting** into manageable chunks (500 characters with 20 overlap)
- **Document preprocessing** for vector storage

### Key Features:
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for document embeddings
- **Vector Store**: Pinecone for efficient similarity search
- **LLM**: Cohere's `command-r-plus` model for response generation
- **Retrieval**: Top-3 similar documents for context

## 🔒 Security Notes

- Keep your API keys secure in the `.env` file
- Never commit the `.env` file to version control
- Consider implementing rate limiting for production use

## 🎯 Customization

### Adjusting Response Length
Modify the word count threshold in the `dynamic_prompt()` function:

```python
if len(query.split()) <= 6:  # Change this threshold
    system_message = short_prompt
```

### Changing Chunk Size
Modify the text splitter parameters in `ingest.py`:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
```

### Adjusting Retrieved Documents
Change the number of retrieved documents in `app.py`:

```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

## 📋 Requirements

The project requires the following main packages:
- `flask` - Web framework
- `langchain` - LLM framework
- `langchain-huggingface` - HuggingFace embeddings
- `langchain-pinecone` - Pinecone integration
- `langchain-cohere` - Cohere LLM integration
- `pinecone-client` - Pinecone vector database
- `python-dotenv` - Environment variable management
- `PyPDF2` or `pypdf` - PDF processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This chatbot is for educational and informational purposes only. Always consult with qualified healthcare professionals for medical advice and treatment decisions.

## 🐛 Troubleshooting

### Common Issues:

1. **Pinecone Connection Error**: Check your API key and index name
2. **Cohere API Error**: Verify your Cohere API key is valid
3. **PDF Loading Issues**: Ensure PDF files are not corrupted and are readable
4. **Memory Issues**: Reduce chunk size or limit the number of documents

### Support

For issues and questions, please open an issue in the repository or contact the maintainers.