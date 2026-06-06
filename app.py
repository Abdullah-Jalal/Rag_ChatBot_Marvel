import chainlit as cl
import asyncio
import os
import sqlite3
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from typing import Optional, Dict
from src.prompt_llm import generate_rag_response

DB_PATH = "chainlit.db"

def init_db(db_path=DB_PATH):
    """
    Initializes a local SQLite database with the tables required by Chainlit's
    SQLAlchemyDataLayer if they do not already exist.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        identifier TEXT NOT NULL UNIQUE,
        metadata TEXT,
        createdAt TEXT
    )
    """)
    
    # Threads Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS threads (
        id TEXT PRIMARY KEY,
        createdAt TEXT,
        name TEXT,
        userId TEXT,
        userIdentifier TEXT,
        tags TEXT,
        metadata TEXT
    )
    """)
    
    # Steps Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS steps (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        threadId TEXT NOT NULL,
        parentId TEXT,
        streaming INTEGER,
        waitForAnswer INTEGER,
        isError INTEGER,
        metadata TEXT,
        tags TEXT,
        input TEXT,
        output TEXT,
        createdAt TEXT,
        start TEXT,
        end TEXT,
        generation TEXT,
        showInput TEXT,
        language TEXT,
        indent INTEGER,
        defaultOpen INTEGER,
        autoCollapse INTEGER,
        modes TEXT
    )
    """)
    
    # Elements Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS elements (
        id TEXT PRIMARY KEY,
        threadId TEXT,
        type TEXT,
        url TEXT,
        chainlitKey TEXT,
        name TEXT NOT NULL,
        display TEXT,
        objectKey TEXT,
        size TEXT,
        page INTEGER,
        language TEXT,
        forId TEXT,
        mime TEXT,
        props TEXT
    )
    """)
    
    # Feedbacks Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedbacks (
        id TEXT PRIMARY KEY,
        forId TEXT NOT NULL,
        threadId TEXT,
        value INTEGER NOT NULL,
        comment TEXT
    )
    """)
    
    conn.commit()
    conn.close()

# Initialize SQLite tables before configuring SQLAlchemyDataLayer
init_db()

# Register the built-in SQLAlchemy data layer with Chainlit
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{DB_PATH}")

# Set up passwordless auto-login for all sessions to enable the thread history sidebar
@cl.header_auth_callback
def header_auth(headers: Dict) -> Optional[cl.User]:
    return cl.User(identifier="marvel_guest", metadata={"role": "user"})

@cl.on_chat_start
async def start():
    # Initialize chat history memory for current session
    cl.user_session.set("message_history", [])
    # Render Marvel-themed custom dashboard welcome screen
    welcome_html = """
    <div class="marvel-dashboard-welcome">
        <div class="marvel-logo-container">
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/Marvel_Logo.svg" alt="Marvel Logo" class="marvel-logo-img" />
        </div>
        <h1 class="hud-title">MARVEL KNOWLEDGE BASE</h1>
        <p class="hud-subtitle">TACTICAL LORE ARCHIVES // SECURE RAG CHATBOT UPLINK</p>
        <div class="hud-status-bar">
            <span class="status-indicator glowing"></span>
            <span class="status-text">SYSTEM STATUS: ALL DATABASE NODES SYNCHRONIZED</span>
        </div>
    </div>
    """
    
    await cl.Message(
        content=welcome_html,
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # 1. Loading State Matching Marvel Dashboard Aesthetics
    msg = cl.Message(content="🔴 *Scanning archives for lore matching query...*")
    await msg.send()

    # Get current session chat history
    message_history = cl.user_session.get("message_history", [])
    history_str = "\n".join(message_history[-6:])  # Keep last 6 interactions

    # 2. Multi-threaded Background RAG Execution
    answer, raw_context = await asyncio.to_thread(generate_rag_response, message.content, history_str)

    # Update history
    message_history.append(f"User: {message.content}")
    message_history.append(f"Assistant: {answer}")
    cl.user_session.set("message_history", message_history)

    # 3. Format and append sources directly to the text response in one go
    sources = []
    for line in raw_context.split('\n'):
        if line.startswith("--- Source ") and line.endswith(" ---"):
            # Extract citation info: "--- Source X: Title (Source) ---"
            source_info = line.strip("- ")
            if ": " in source_info:
                source_info = source_info.split(": ", 1)[1]
            if source_info not in sources:
                sources.append(source_info)

    final_content = answer
    if sources:
        final_content += "\n\n---\n### 📚 Sources\n" + "\n".join(f"- {s}" for s in sources)

    # 4. Return Final Response
    msg.content = final_content
    await msg.update()