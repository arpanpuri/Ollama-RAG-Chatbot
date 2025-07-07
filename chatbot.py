import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot (Ollama)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        llm = OllamaLLM(model="llama2")
        
        groq_sys_prompt = PromptTemplate.from_template(
            """You are very smart at everything, you always give the best, the most accurate and most precise answer. Answer the following Question: {user_prompt}. Start the answer directly. No small talk please"""
        )
        
        chain = groq_sys_prompt | llm | StrOutputParser()
        response = chain.invoke({"user_prompt": prompt})
        
    except Exception as e:
        response = f"Error: {e}. Make sure Ollama is running and you have the llama2 model installed."

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("About")
    st.write("This is a RAG (Retrieval-Augmented Generation) chatbot using Ollama.")
    st.write("Make sure Ollama is running and you have the llama2 model installed.")
    st.write("To install llama2: ollama pull llama2")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()



        