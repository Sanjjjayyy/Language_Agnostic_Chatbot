# Conversational RAG Chatbot

This project is a complete, end-to-end Conversational Retrieval-Augmented Generation (RAG) system. It allows you to build a sophisticated chatbot that can answer questions based on a private collection of PDF documents. The entire pipeline, from data ingestion to a user-friendly web interface, is included.

## Features

* **End-to-End Data Pipeline**: A series of scripts to automatically process raw documents and prepare them for the RAG system.
    * **Advanced Document Ingestion**: Extracts text and tables from PDFs, with an automatic OCR fallback for scanned documents.
    * **Intelligent Text Cleaning**: A multi-step cleaning process to normalize text, remove noise, and filter out low-quality content.
    * **Content-Aware Chunking**: Splits documents into meaningful chunks, keeping tables intact and filtering out "stub" chunks.
    * **Incremental Processing**: All pipeline steps are incremental, only processing new or updated files to save time and resources.
* **Conversational AI Core**:
    * **State-of-the-Art RAG Chain**: Uses a modern, conversational RAG chain that remembers chat history to answer follow-up questions.
    * **Powered by Gemini**: Leverages Google's Gemini Pro for high-quality, context-aware answer generation.
    * **Source-Cited Answers**: The chatbot returns the source documents it used to generate an answer, providing transparency and trust.
* **Web Interface**:
    * **FastAPI Backend**: A robust and efficient API to serve the RAG chain.
    * **Streamlit Frontend**: A user-friendly, interactive chat interface for easy interaction with the chatbot.

## Tech Stack

* **Backend**: Python, FastAPI
* **Frontend**: Streamlit
* **LLM**: Google Gemini 1.5 Flash via `langchain-google-genai`
* **Embeddings**: `intfloat/multilingual-e5-large`
* **Vector Store**: FAISS 
* **Orchestration**: LangChain
* **Data Processing**: PyPDF2, pdfplumber, pytesseract, python-docx



## Project Structure

```bash
├── api/
│   └── app.py                # FastAPI application
├── chunks/                   # Stores the chunked documents
├── data/                     # Raw PDF documents go here
├── embeddings/ 
│   ├── create_embeddings.py  # Script to generate embeddings
│   └── load_to_faiss.py      # Loads the Embeddings into FAISS
├── frontend/
│   └── app.py                # Streamlit frontend application
├── ingestion/
│   └── load_documents.py     # Script to ingest and clean documents
├── processing/
│   └── chunks_documents.py   # Script to chunk the cleaned documents
├── query/
│   └── query_faiss.py        # CLI tool to test FAISS index
├── rag/
│   ├── config.py             # Configuration for the RAG chain
│   ├── memory_buffer.py      # Manages conversational memory
│   └── rag_chain.py          # Main RAG chain 
├── .env                      # Store API keys (need to be created)
└── .gitignore                # Git ignore rules
```


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/Sanjjjayyy/Language_Agnostic_Chatbot>
    cd <Language Agnostic Chatbot>
    ```

2.  **Set up a Conda environment:**
    ```bash
    conda create --name rag-chatbot python=3.10
    conda activate rag-chatbot
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create the `.env` file:**
    In the root directory of the project, create a file named `.env` and add your API key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

##  Usage

The project is divided into two main parts: the data pipeline and the live application.

### Part 1: Running the Data Pipeline

You must run these scripts in order from the root directory.

1.  **Add Documents**: Place all your PDF and DOCX files into the `data/` folder.

2.  **Ingest and Clean Documents**:
    ```bash
    python -m ingestion.load_documents
    ```

3.  **Chunk the Documents**:
    ```bash
    python -m processing.chunks_documents
    ```

4.  **Create Embeddings**:
    ```bash
    python -m embeddings.create_embeddings
    ```

5.  **Build the FAISS Index**:
    ```bash
    python -m embeddings.load_to_faiss
    ```
    After this step, your knowledge base is ready.

### Part 2: Running the Chatbot Application

This requires two separate terminals, both running from the project's root directory.

**Terminal 1: Start the Backend API**
```bash
conda activate rag-chatbot
uvicorn api.app:app --reload --port 8000
```

Wait for the message indicating the application startup is complete.

**Terminal 2: Start the Frontend UI**
```bash
conda activate rag-chatbot
streamlit run frontend/app.py
```
This will open a new tab in your web browser with the chatbot interface. You can now start asking questions.