import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

def run_diagnostics():
    print("==================================================")
    print("🔒 STARTING API KEY & CONNECTION DIAGNOSTICS")
    print("==================================================\n")
    
    # 1. Load Environment Variables
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    
    # 2. Check Groq Key Presence
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("❌ CRITICAL: GROQ_API_KEY is missing or unconfigured in .env")
        return
    else:
        print(f"✅ GROQ_API_KEY found: {groq_key[:8]}...{groq_key[-4:]}")
        
    # 3. Check Hugging Face Token Presence
    if not hf_token or hf_token == "your_huggingface_token_here":
        print("⚠️  WARNING: HF_TOKEN is not configured. Running in unauthenticated public mode.")
    else:
        print(f"✅ HF_TOKEN found: {hf_token[:6]}...{hf_token[-4:]}")

    print("\n--------------------------------------------------")
    print("🧠 TESTING HUGGING FACE EMBEDDINGS FRAMEWORK")
    print("--------------------------------------------------")
    
    try:
        print("🔄 Initializing 'all-MiniLM-L6-v2' local pipeline...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Run a quick mathematical vector translation test
        test_vector = embeddings.embed_query("Marvel Cinematic Universe")
        print(f"✅ Embeddings initialized successfully!")
        print(f"📐 Vector Dimensions Generated: {len(test_vector)} (Expected: 384)")
    except Exception as e:
        print(f"❌ Hugging Face Initialization Failed: {e}")
        return

    print("\n--------------------------------------------------")
    print("🚀 TESTING GROQ CLOUD INFERENCE CONNECTION")
    print("--------------------------------------------------")
    
    try:
        print("🔄 Sending live ping request to Llama-3 hardware cluster...")
        llm = ChatGroq(
            temperature=0.0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_key
        )
        
        # Standard system validation query
        response = llm.invoke("Respond with exactly the phrase: 'Groq Connection Successful.'")
        print(f"✅ Groq Live API Response: {response.content.strip()}")
    except Exception as e:
        print(f"❌ Groq API Connection Failed!")
        print(f"📝 Error Details: {e}")
        print("\n💡 Tip: Double-check your Groq dashboard to ensure your API key hasn't expired or been deleted.")
        return

    print("\n==================================================")
    print("🎉 ALL SYSTEMS OPERATIONAL: READY FOR INGESTION & PIPELINE RUNS")
    print("==================================================")

if __name__ == "__main__":
    run_diagnostics()