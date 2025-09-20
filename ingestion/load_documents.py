# ingestion/load_documents.py
import os, glob, json, re, sys
from PyPDF2 import PdfReader
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import docx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_FOLDER, PROCESSED_DOCS_FOLDER

PROCESSED_FILES_TRACKER = os.path.join(PROCESSED_DOCS_FOLDER, "processed_files.json")

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_TRACKER):
        with open(PROCESSED_FILES_TRACKER, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_set):
    os.makedirs(PROCESSED_DOCS_FOLDER, exist_ok=True)
    with open(PROCESSED_FILES_TRACKER, "w") as f:
        json.dump(list(processed_set), f)

def save_chunks_to_json(filename, chunks):
    os.makedirs(PROCESSED_DOCS_FOLDER, exist_ok=True)
    out_path = os.path.join(PROCESSED_DOCS_FOLDER, filename + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_likely_text(text, alpha_ratio=0.5, min_length=10):
    if not text or len(text.strip()) < min_length: return False
    alpha_chars = sum(1 for char in text if char.isalpha())
    if len(text.strip()) > 0 and (alpha_chars / len(text.strip())) < alpha_ratio: return False
    return True

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    chunks = []
    filename = os.path.basename(file_path)

    if ext == ".pdf":
        raw_text_content = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = "\n".join([" | ".join(filter(None, cell.split('\n')) if cell else '' for cell in row) for row in table])
                            chunks.append({"content": clean_text(table_text), "metadata": {"source": filename, "page": i, "type": "table"}})
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if page_text:
                        raw_text_content += page_text + "\n\n"
            if not chunks and not raw_text_content.strip():
                images = convert_from_path(file_path)
                for i, img in enumerate(images, start=1):
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                         chunks.append({"content": clean_text(ocr_text), "metadata": {"source": filename, "page": i, "type": "ocr"}})
        except Exception:
            try:
                images = convert_from_path(file_path)
                for i, img in enumerate(images, start=1):
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                         chunks.append({"content": clean_text(ocr_text), "metadata": {"source": filename, "page": i, "type": "ocr"}})
            except Exception:
                pass
        if raw_text_content:
            paragraphs = raw_text_content.split('\n\n')
            for para in paragraphs:
                cleaned_para = clean_text(para)
                if is_likely_text(cleaned_para):
                    chunks.append({"content": cleaned_para, "metadata": {"source": filename, "type": "text"}})
    elif ext == ".docx":
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    chunks.append({"content": clean_text(para.text), "metadata": {"source": filename, "type": "text"}})
        except Exception:
            pass
    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            if full_text.strip():
                chunks.append({"content": clean_text(full_text), "metadata": {"source": filename, "type": "text"}})
        except Exception:
            pass
    return chunks

def load_new_documents():
    processed_files = load_processed_files()
    new_files_set = set()
    files_found = glob.glob(os.path.join(DATA_FOLDER, "*"))
    for file_path in files_found:
        filename = os.path.basename(file_path)
        if filename in processed_files: continue
        chunks = extract_text_from_file(file_path)
        if chunks:
            save_chunks_to_json(filename, chunks)
            new_files_set.add(filename)
    if new_files_set:
        processed_files.update(new_files_set)
        save_processed_files(processed_files)

if __name__ == "__main__":
    load_new_documents()