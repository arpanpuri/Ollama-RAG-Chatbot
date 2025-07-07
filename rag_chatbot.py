import streamlit as st
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Status cards */
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-card.warning {
        border-left-color: #ffc107;
    }
    
    .status-card.error {
        border-left-color: #dc3545;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f0f2ff;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background: #111;
        border: 1px solid #222;
        margin-right: 2rem;
        color: #fff;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Welcome section */
    .welcome-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Progress indicators */
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Source citations */
    .sources-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid #17a2b8;
    }
    
    .source-item {
        background: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "processing_status" not in st.session_state:
    st.session_state.processing_status = ""

st.markdown("""
<div class="main-header">
    <h1>🤖 AI RAG Assistant</h1>
    <p>Your intelligent document analysis and conversation partner</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 Document Management")
    
    st.markdown('<div class="sidebar .sidebar-content">', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📁 Upload Documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or text files to create your knowledge base"
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} file(s) selected")
        
        for file in uploaded_files:
            st.info(f"📄 {file.name} ({file.size} bytes)")
    
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        with st.spinner("🔄 Processing your documents..."):
            try:
                st.session_state.processing_status = "Processing documents..."
                
                embeddings = OllamaEmbeddings(model="llama2")
                
                documents = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        if uploaded_file.name.endswith('.pdf'):
                            loader = PyPDFLoader(tmp_file_path)
                        elif uploaded_file.name.endswith('.txt'):
                            loader = TextLoader(tmp_file_path)
                        else:
                            continue
                        
                        documents.extend(loader.load())
                        
                    finally:
                        os.unlink(tmp_file_path)
                
                if documents:
                    st.session_state.processing_status = "Creating embeddings..."
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    st.session_state.processing_status = "Building vector database..."
                    
                    
                    st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
                    st.session_state.documents_loaded = True
                    st.session_state.processing_status = ""
                    
                    st.success(f"✅ Successfully processed {len(documents)} documents!")
                    st.info(f"📊 Created {len(splits)} knowledge chunks")
                else:
                    st.error("❌ No valid documents found!")
                    
            except Exception as e:
                st.error(f"❌ Error processing documents: {e}")
                st.session_state.processing_status = ""
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.session_state.documents_loaded:
            if st.button("📚 Clear Docs", type="secondary"):
                st.session_state.vectorstore = None
                st.session_state.documents_loaded = False
                st.session_state.messages = []
                st.rerun()

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <strong>You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <strong>AI Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

if prompt := st.chat_input("💬 Ask me anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        llm = OllamaLLM(model="llama2")
        
        if st.session_state.documents_loaded and st.session_state.vectorstore:
            with st.spinner("🔍 Searching documents..."):
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                
                result = qa_chain.invoke({"query": prompt})
                response = result["result"]
                
                if result.get("source_documents"):
                    response += "\n\n**📖 Sources:**\n"
                    for i, doc in enumerate(result["source_documents"][:2], 1):
                        source_name = doc.metadata.get('source', 'Unknown source')
                        response += f"{i}. {source_name}\n"
        else:
            with st.spinner("🤔 Thinking..."):
                prompt_template = PromptTemplate.from_template(
                    """You are a helpful AI assistant. Answer the following question: {question}"""
                )
                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({"question": prompt})
        
    except Exception as e:
        response = f"❌ Error: {e}. Make sure Ollama is running and you have the llama2 model installed."

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if not st.session_state.messages:
    pass

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 Powered by Ollama & LangChain | 🔒 Your data stays local</p>
</div>
""", unsafe_allow_html=True) 