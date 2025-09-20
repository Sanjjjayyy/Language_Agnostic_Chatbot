# processing/chunks_documents.py
import os, glob, json, sys
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROCESSED_DOCS_FOLDER, CHUNKS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

PROCESSED_CHUNKED = os.path.join(CHUNKS_FOLDER, "processed_chunked.json")

def load_processed_chunked():
    if os.path.exists(PROCESSED_CHUNKED):
        with open(PROCESSED_CHUNKED, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_chunked(processed_set):
    os.makedirs(CHUNKS_FOLDER, exist_ok=True)
    with open(PROCESSED_CHUNKED, "w") as f:
        json.dump(list(processed_set), f)

def save_chunks(filename, chunks):
    os.makedirs(CHUNKS_FOLDER, exist_ok=True)
    out_path = os.path.join(CHUNKS_FOLDER, filename + "_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def chunk_document(doc_json):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
    final_chunks = []
    for entry in doc_json:
        content = entry.get("content", "").strip()
        metadata = entry.get("metadata", {})
        if not content: continue
        if metadata.get("type") == "table":
            final_chunks.append({"content": content, "metadata": {**metadata, "chunk_id": 1}})
            continue
        splits = text_splitter.split_text(content)
        for i, chunk in enumerate(splits, start=1):
            if len(chunk) >= MIN_CHUNK_SIZE:
                final_chunks.append({"content": chunk, "metadata": {**metadata, "chunk_id": i}})
    return final_chunks

if __name__ == "__main__":
    processed_chunked = load_processed_chunked()
    new_chunked = set()
    files = glob.glob(os.path.join(PROCESSED_DOCS_FOLDER, "*.json"))
    for file_path in files:
        fname = os.path.basename(file_path).replace(".json", "")
        if fname in processed_chunked: continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc_json = json.load(f)
            if not isinstance(doc_json, list) or not all(isinstance(entry, dict) and 'content' in entry for entry in doc_json): continue
            chunks = chunk_document(doc_json)
            if chunks:
                save_chunks(fname, chunks)
                new_chunked.add(fname)
        except Exception:
            pass
    if new_chunked:
        processed_chunked.update(new_chunked)
        save_processed_chunked(processed_chunked)