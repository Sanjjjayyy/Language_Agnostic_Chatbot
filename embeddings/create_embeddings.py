# embeddings/create_embeddings.py
import os, json, sys
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CHUNKS_FOLDER, EMBEDDINGS_FOLDER, EMBEDDING_MODEL_NAME

MODEL_SUBFOLDER = EMBEDDING_MODEL_NAME.split('/')[-1]
MODEL_EMBEDDINGS_FOLDER = os.path.join(EMBEDDINGS_FOLDER, MODEL_SUBFOLDER)
os.makedirs(MODEL_EMBEDDINGS_FOLDER, exist_ok=True)

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_existing_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_embeddings(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunk_files = [f for f in os.listdir(CHUNKS_FOLDER) if f.endswith('_chunks.json')]
    for chunk_file in chunk_files:
        chunk_path = os.path.join(CHUNKS_FOLDER, chunk_file)
        embedding_path = os.path.join(MODEL_EMBEDDINGS_FOLDER, chunk_file.replace('_chunks.json', '_embeddings.json'))
        chunks = load_chunks(chunk_path)
        existing_embeddings = load_existing_embeddings(embedding_path)
        processed_keys = {f"{item['metadata'].get('source')}|{hash(item.get('content',''))}" for item in existing_embeddings}
        new_data = []
        for chunk in chunks:
            text = chunk["content"].strip()
            if not text: continue
            key = f"{chunk['metadata'].get('source')}|{hash(text)}"
            if key in processed_keys: continue
            new_data.append(chunk)
        if not new_data: continue
        texts = [c["content"] for c in new_data]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        for emb, meta, txt in zip(embeddings, [c["metadata"] for c in new_data], texts):
            existing_embeddings.append({"embedding": emb.tolist(), "metadata": meta, "content": txt})
        save_embeddings(existing_embeddings, embedding_path)

if __name__ == "__main__":
    main()