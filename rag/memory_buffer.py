# rag/memory_buffer.py
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True).chat_memory
    return store[session_id]