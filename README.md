# Ollama-RAG-Chatbot

A modern, private, and powerful Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Ollama. Upload your own documents (PDF/TXT) and chat with your data using local LLMs—no API keys required!

## Features
- **RAG (Retrieval-Augmented Generation)**: Ask questions about your own documents
- **Ollama LLM**: Uses local models (e.g., Llama2) for privacy and speed
- **Streamlit UI**: Beautiful, modern, and interactive web interface
- **Document Upload**: Supports PDF and TXT files
- **Semantic Search**: Finds relevant information in your documents
- **Source Citations**: See where answers come from
- **No API Keys Needed**: 100% local, works offline

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/arpanpuri/Ollama-RAG-Chatbot.git
   cd Ollama-RAG-Chatbot
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install faiss-cpu
   pip install langchain-ollama
   ```
3. **Start Ollama server** (in a separate terminal)
   ```bash
   ollama serve
   ollama pull llama2
   ```
4. **Run the chatbot**
   ```bash
   streamlit run rag_chatbot.py
   ```

## Usage
- Upload PDF or TXT files in the sidebar
- Click "Process Documents" to create a knowledge base
- Ask questions in the chat box
- Get answers with source citations

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Streamlit, LangChain, FAISS, langchain-ollama

---

> **Your data stays private. All processing is local.** 