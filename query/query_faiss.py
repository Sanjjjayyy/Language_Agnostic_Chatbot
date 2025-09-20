# query/query_faiss.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- CONFIG ----------
MODEL_NAME = "intfloat/multilingual-e5-large"
MODEL_SUBFOLDER = MODEL_NAME.split('/')[-1]

# The correct, updated path to your FAISS index
FAISS_INDEX_PATH = os.path.join("../embeddings", MODEL_SUBFOLDER, "faiss_index")

# Number of search results to return
K = 5

# ---------- FUNCTIONS ----------

def search(query, vectorstore):
    """
    Performs a similarity search on the vectorstore and returns the top K results.
    """
    print(f"\nSearching for: '{query}'...")
    results = vectorstore.similarity_search(query, k=K)
    return results

def display_results(results):
    """
    Prints the search results in a user-friendly format.
    """
    if not results:
        print("No results found.")
        return

    print("\n--- Top Results ---")
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Content: {doc.page_content}")
        # Print the rich metadata
        if doc.metadata:
            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}")
            print(f"  Type: {doc.metadata.get('type', 'N/A')}")
    print("\n--------------------")


# ---------- MAIN ----------

def main():
    # Check if the FAISS index exists
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"[ERROR] FAISS index not found at the specified path: {FAISS_INDEX_PATH}")
        print("Please run the `load_to_faiss.py` script first to create the index.")
        return

    # Load the embedding model and the FAISS index
    print("[INFO] Loading embedding model and FAISS index...")
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings_model,
            # This is required for loading FAISS indexes created with older LangChain versions
            allow_dangerous_deserialization=True 
        )
        print("✅ Index loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load the FAISS index: {e}")
        return

    # Start the interactive query loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue

        results = search(query, vectorstore)
        display_results(results)


# ---------- RUN SCRIPT ----------
if __name__ == "__main__":
    main()