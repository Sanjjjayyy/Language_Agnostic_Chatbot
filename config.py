# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- PATHS ---
# Dynamically calculate the project's root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Data Pipeline Folders ---
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DOCS_FOLDER = os.path.join(PROJECT_ROOT, "processed_docs")
CHUNKS_FOLDER = os.path.join(PROJECT_ROOT, "chunks")
EMBEDDINGS_FOLDER = os.path.join(PROJECT_ROOT, "embeddings")


# --- MODEL CONFIGURATION ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_NAME = "gemini-1.5-flash"


# --- VECTOR STORE CONFIGURATION ---
MODEL_SUBFOLDER = EMBEDDING_MODEL_NAME.split('/')[-1]
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_FOLDER, MODEL_SUBFOLDER, "faiss_index")


# --- DATA PROCESSING CONFIGURATION ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50


# --- RAG CHAIN CONFIGURATION ---
K = 5 # Number of search results to return
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 1024


# --- API & FRONTEND CONFIGURATION ---
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}/chat"