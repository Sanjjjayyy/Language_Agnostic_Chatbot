# api/app.py
import sys, os
from fastapi import FastAPI, Form
from pydantic import BaseModel
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import API_HOST, API_PORT
from rag.rag_chain import load_vectorstore, create_rag_chain, get_answer

app = FastAPI(title="Conversational RAG API")
vectorstore = load_vectorstore()
rag_chain = create_rag_chain(vectorstore)

class Source(BaseModel):
    content: str
    metadata: Dict

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.post("/chat", response_model=ChatResponse)
async def chat(query: str = Form(...), session_id: str = Form("default_session")):
    response = get_answer(query, session_id, rag_chain)
    source_documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in response.get("context", [])]
    return {"answer": response["answer"], "sources": source_documents}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)