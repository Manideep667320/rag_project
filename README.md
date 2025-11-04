# RAG Web Application

A Retrieval-Augmented Generation (RAG) web application built with Flask. This application allows users to upload documents, process them, and query the content using natural language.

## Features

- Document upload and processing (supports PDF, DOCX, and text files)
- Real-time document processing with progress tracking
- Conversational Q&A interface
- Background file system monitoring for automatic updates
- Support for multiple document types
- Conversation memory for contextual responses

## Prerequisites

- Python 3.8+
- pip package manager
- OpenAI API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag_project
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

1. **Start the web application**
   ```bash
   python run.py
   ```

2. **Access the application**
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

- `src/` - Main application source code
  - `web.py` - Flask web application and routes
  - `ingest.py` - Document processing and vector store management
  - `rag_engine.py` - RAG pipeline implementation
  - `conversation_manager.py` - Conversation memory handling
- `templates/` - HTML templates for the web interface
- `static/` - Static files (CSS, JavaScript, etc.)
- `data/` - Directory for storing uploaded documents
- `embeddings/` - Vector store and document embeddings
- `tests/` - Test files

## Usage

1. **Upload Documents**
   - Click the "Upload Files" button to add documents
   - Supported formats: PDF, DOCX, TXT
   - Uploaded files will be automatically processed

2. **Ask Questions**
   - Type your question in the chat interface
   - The system will retrieve relevant information from your documents
   - Responses are generated using the RAG model

3. **Manage Conversations**
   - Use the "Clear Chat" button to start a new conversation
   - The system maintains conversation context for follow-up questions

## Configuration

The application can be configured using environment variables in the `.env` file:

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `UPLOAD_FOLDER` - Directory for storing uploaded files (default: `data/raw`)
- `VECTORSTORE_PATH` - Path to store the vector database (default: `embeddings/vectorstore`)

## Testing

Run the test suite using pytest:

```bash
pytest -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
