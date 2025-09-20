# embeddings/load_to_faiss.py
import os, json, sys
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, EMBEDDINGS_FOLDER

MODEL_SUBFOLDER = EMBEDDING_MODEL_NAME.split('/')[-1]
MODEL_EMBEDDINGS_FOLDER = os.path.join(EMBEDDINGS_FOLDER, MODEL_SUBFOLDER)
PROCESSED_FILES_TRACKER = os.path.join(MODEL_EMBEDDINGS_FOLDER, "processed_in_faiss.json")

def load_processed_files_tracker():
    if os.path.exists(PROCESSED_FILES_TRACKER):
        with open(PROCESSED_FILES_TRACKER, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_processed_files_tracker(processed_files):
    with open(PROCESSED_FILES_TRACKER, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, indent=2)

def prepare_new_data(processed_files):
    new_files = {f for f in os.listdir(MODEL_EMBEDDINGS_FOLDER) if f.endswith("_embeddings.json")} - processed_files
    if not new_files: return None, None, None, None
    texts, embeddings, metadatas = [], [], []
    for fname in new_files:
        fpath = os.path.join(MODEL_EMBEDDINGS_FOLDER, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    texts.append(item["content"])
                    embeddings.append(item["embedding"])
                    metadatas.append(item["metadata"])
        except (json.JSONDecodeError, KeyError):
            pass
    return texts, embeddings, metadatas, new_files

def main():
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    processed_files = load_processed_files_tracker()
    texts, embeddings, metadatas, new_files = prepare_new_data(processed_files)
    if not texts: return
    text_embedding_pairs = list(zip(texts, embeddings))
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
        vectorstore.add_embeddings(text_embedding_pairs, metadatas=metadatas)
    else:
        vectorstore = FAISS.from_embeddings(text_embedding_pairs, embeddings_model, metadatas=metadatas)
    vectorstore.save_local(FAISS_INDEX_PATH)
    processed_files.update(new_files)
    save_processed_files_tracker(processed_files)

if __name__ == "__main__":
    main()