# rag/rag_chain.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, LLM_NAME, TEMPERATURE, K
from rag.memory_buffer import get_session_history

def load_vectorstore():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}.")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)

def create_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model=LLM_NAME, temperature=TEMPERATURE)
    retriever = vectorstore.as_retriever(search_kwargs={'k': K})
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an college assistant bot for Francis Xavier Engineering College. Use only the pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know politely and ask the user to contact their mentor. Use three sentences maximum and keep the answer concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_answer(question, session_id, rag_chain):
    chat_history = get_session_history(session_id)
    response = rag_chain.invoke({"input": question, "chat_history": chat_history.messages})
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response["answer"])
    return response

if __name__ == '__main__':
    vectorstore = load_vectorstore()
    rag_chain = create_rag_chain(vectorstore)
    session_id = "test_session_cli"
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit': break
        if not query.strip(): continue
        response = get_answer(query, session_id, rag_chain)
        print(f"\nBot: {response['answer']}")
        print("\n--- Sources Used ---")
        for doc in response.get("context", []):
            print(f"  - Source: {doc.metadata.get('source', 'N/A')}")