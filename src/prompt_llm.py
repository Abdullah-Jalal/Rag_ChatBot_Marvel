import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.retrieve import retrieve_context

# 1. Load environment variables from the root .env file
load_dotenv()

def generate_rag_response(user_query, chat_history_str=""):
    # 2. Retrieve the matching Wikipedia context chunks using your retrieve script
    context = retrieve_context(user_query, k=4)
    
    if not context.strip():
        return "The local vector database appears to be empty or failed to load.", context

    # 3. Initialize the Groq inference client using your verified Llama 3.1 model
    try:
        llm = ChatGroq(
            temperature=0.0,  # Strict zero-temperature to ensure deterministic, factual output
            model_name="llama-3.1-8b-instant", 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    except Exception as e:
        return f"[ERROR] Error initializing Groq LLM. Check your GROQ_API_KEY. Details: {e}", context

    # 4. Construct a strict System Prompt to prevent hallucinations (as mandated by syllabus)
    system_prompt = (
        "You are an objective AI assistant specialized in Marvel lore. Synthesize a comprehensive response "
        "to the user query using ONLY the verified source context fragments provided below.\n\n"
        "CRITICAL RULES:\n"
        "1. If the answer cannot be confidently deduced directly from the provided text context, reply "
        "explicitly with: 'The requested target info is missing from the provided dataset.'\n"
        "2. Do not extrapolate, assume, or use external knowledge outside of the provided context.\n"
        "3. Keep your response factual and professional.\n\n"
        "PREVIOUS CHAT HISTORY (Use this to understand context if the user asks follow-up questions):\n{chat_history}\n\n"
        "[CONTEXT]:\n{retrieved_chunks}"
    )

    # 5. Compile the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "[USER QUERY]: {user_query}")
    ])

    # 6. Chain the elements together and invoke the inference hardware
    chain = prompt_template | llm
    
    print(f"Dispatching context-injected prompt to Groq API (llama-3.1-8b-instant)...")
    response = chain.invoke({
        "retrieved_chunks": context,
        "chat_history": chat_history_str,
        "user_query": user_query
    })
    
    # 7. Return BOTH the generated text answer AND the raw context sources for the UI
    return response.content, context

if __name__ == "__main__":
    # A quick terminal test to ensure the LLM is reading the database correctly
    query = "What is the Marvel Cinematic Universe?"
    answer, sources = generate_rag_response(query)
    
    print("\n==================================================")
    print("AI ANSWER:")
    print("==================================================\n")
    print(answer)