import os
import time
import wikipedia  
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Maintain API compliance
wikipedia.set_user_agent("MarvelRagChatbot/2.0_Enterprise (academic_evaluation_project@example.com)")

DB_PATH = "vector_db"

def ingest_and_store():
    # THE MASTER LORE MATRIX
    # We target dense "List" pages and massive crossover events to maximize data density
    topics = [
        # MCU Complete
        "Marvel Cinematic Universe", "List of Marvel Cinematic Universe films", 
        "List of Marvel Cinematic Universe television series", "Timeline of the Marvel Cinematic Universe",
        
        # Massive Catalogs & Indexes
        "List of Marvel Comics characters", "List of Marvel Comics teams and organizations",
        "List of alien races in Marvel Comics", "List of deities in Marvel Comics",
        "List of Marvel Comics superhero debuts", "Features of the Marvel Universe",
        
        # Major Crossover Events (The core lore of the comics)
        "Secret Wars (1984 comic book)", "Secret Wars (2015 comic book)",
        "Infinity Gauntlet", "Civil War (comics)", "House of M", "Avengers vs. X-Men",
        "Age of Ultron", "Secret Invasion", "Annihilation (comics)", "Spider-Verse",

        # Cosmic Entities & Artifacts
        "Eternals (comics)", "Celestials (comics)", "Watchers (Marvel Comics)",
        "Cosmic Cube", "Infinity Gems", "Ultimate Nullifier", "Vibranium", "Adamantium",

        # Deep Lore Teams
        "Avengers (comics)", "X-Men", "Fantastic Four", "Guardians of the Galaxy (1969 team)",
        "Guardians of the Galaxy (2008 team)", "Inhumans", "Defenders (comics)", 
        "Illuminati (comics)", "Midnight Sons", "Thunderbolts (comics)", "Sinister Six",
        
        # Key Locations
        "Wakanda", "Asgard (comics)", "Latveria", "Savage Land", "Madripoor", "Krakoa"
    ]
    
    raw_documents = []
    print(f"🛡️ [SYSTEM] Initiating Deep-Lore Data Lake Ingestion. Total Targets: {len(topics)}")
    
    # 2. Rate-Limited Ingestion Loop
    for i, topic in enumerate(topics):
        try:
            # We increase load_max_docs to pull secondary connected pages, and increase character limits
            loader = WikipediaLoader(query=topic, load_max_docs=3, doc_content_chars_max=40000)
            docs = loader.load()
            raw_documents.extend(docs)
            print(f"  [+] {i+1}/{len(topics)} Downloaded: '{topic}' ({len(docs)} files)")
            
            # CRITICAL: Pause for 1.5 seconds between requests to prevent API bans
            time.sleep(1.5) 
            
        except Exception as e:
            print(f"  [!] Failed to load '{topic}': {e}")

    print(f"\n📊 [DATA] Total raw documents loaded into memory: {len(raw_documents)}")

    if not raw_documents:
        print("❌ [ERROR] Ingestion halted. No data retrieved.")
        return

    # 3. Chunking 
    print("✂️ [PROCESSING] Slicing data into 500-token vectors...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunked_docs = text_splitter.split_documents(raw_documents)
    print(f"📦 [DATA] Created {len(chunked_docs)} distinct vector-ready text chunks.")

    # 4. Neural Embedding
    print("🧠 [NEURAL] Initializing Hugging Face Math Engine (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # 5. Database Compilation
    print("💾 [DATABASE] Compiling FAISS index. This may take a moment...")
    try:
        vector_store = FAISS.from_documents(chunked_docs, embeddings)
        vector_store.save_local(DB_PATH)
        print(f"✅ [SUCCESS] Massive Vector Database compiled and secured in '{DB_PATH}'!")
    except Exception as e:
        print(f"❌ [ERROR] FAISS compilation failed: {e}")

if __name__ == "__main__":
    ingest_and_store()