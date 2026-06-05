import os
import wikipedia  # Import raw library to configure global API client settings
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. CRITICAL FIX: Set a custom User-Agent to satisfy Wikimedia's API access policy.
# This prevents their servers from blocking your IP with a 403 HTML error.
wikipedia.set_user_agent("MarvelRagChatbot/1.0 (academic_evaluation_project@example.com)")

DB_PATH = "vector_db"

def ingest_and_store():
    # Define targeted, dense Wikipedia topics for the dataset
    topics = [
        "Marvel Cinematic Universe",
        "List of Marvel Cinematic Universe films",
        "Avengers (comics)"
    ]
    
    raw_documents = []
    print("🤖 Starting data ingestion from Wikipedia...")
    
    # 2. Content Ingestion: Cleanly load text and strip bad HTML markers
    for topic in topics:
        try:
            loader = WikipediaLoader(query=topic, load_max_docs=2, doc_content_chars_max=20000)
            docs = loader.load()
            raw_documents.extend(docs)
            print(f"✅ Successfully loaded context for: '{topic}'")
        except Exception as e:
            print(f"❌ Failed to load '{topic}': {e}")

    print(f"\nTotal raw documents loaded: {len(raw_documents)}")

    # 3. Defensive Guard Clause: Stop execution cleanly if no data was fetched
    if not raw_documents:
        print("⚠️  CRITICAL ERROR: No documents were retrieved from the API. Ingestion halted.")
        return

    # 4. Token Chunking: 500-token chunks with a 10% overlap (50 tokens)
    print("🧩 Chunking text into strict 500-token blocks...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo", # Uses standard cl100k_base tokenizer
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunked_docs = text_splitter.split_documents(raw_documents)
    print(f"🧩 Created {len(chunked_docs)} distinct vector-ready text chunks.")

    # 5. Mathematical Vector Generation
    print("🧠 Initializing Hugging Face embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 6. Initialize a dedicated Vector Database and Persistence
    print("💾 Generating vector matrices and saving to FAISS database...")
    try:
        vector_store = FAISS.from_documents(chunked_docs, embeddings)
        # Save the index locally so the retrieval script can load it later
        vector_store.save_local(DB_PATH)
        print(f"✅ Vector database successfully initialized and saved to the '{DB_PATH}' folder!")
    except Exception as e:
        print(f"❌ FAISS Vector Store creation failed: {e}")

if __name__ == "__main__":
    ingest_and_store()