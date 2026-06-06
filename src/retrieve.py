import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define the local path where we saved the database
DB_PATH = "vector_db"

def retrieve_context(user_query, k=4):
    print(f"Searching Marvel knowledge base for: '{user_query}'...")
    
    # 1. Initialize the exact same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    
    # 2. Load the local FAISS vector index
    try:
        # allow_dangerous_deserialization is required to load local pickle files safely in LangChain
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[ERROR] Error loading vector database. Did you run ingest.py first? Details: {e}")
        return ""

    # 3. Fetch the top 'k' most statistically relevant text blocks based on distance calculations
    retrieved_docs = vector_store.similarity_search(user_query, k=k)
    
    print(f"[SUCCESS] Successfully fetched top {len(retrieved_docs)} context chunks.\n")
    
    # 4. Format the retrieved chunks into a single string so we can feed it to the LLM later
    formatted_context = ""
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get('source', 'Unknown Wikipedia Source')
        title = doc.metadata.get('title', 'Unknown Title')
        
        # We append the metadata to the text so the final UI can cite its sources
        formatted_context += f"--- Source {i+1}: {title} ({source}) ---\n"
        formatted_context += f"{doc.page_content}\n\n"
        
    return formatted_context

if __name__ == "__main__":
    # A quick local test to verify our retrieval pipeline works
    sample_query = "Who are the original founding members of the Avengers?"
    test_result = retrieve_context(sample_query)
    
    print("==================================================")
    print("RAW CONTEXT RETRIEVED FROM DATABASE:")
    print("==================================================\n")
    print(test_result)