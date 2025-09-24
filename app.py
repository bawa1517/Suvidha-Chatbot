import streamlit as st
import pdfplumber
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import os
import time
import base64
import json
from datetime import datetime
from PIL import Image

# === Streamlit UI Setup ===
st.set_page_config(
    page_title="📄 SJVNL Suvidha Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Function to encode image to base64 ===
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# === Load and encode SJVN logo ===
def load_sjvn_logo():
    try:
        # Try to load the SJVN.png image
        if os.path.exists("SJVN.png"):
            return get_base64_of_bin_file("SJVN.png")
        else:
            return None
    except:
        return None

# Get the base64 encoded image
sjvn_logo_base64 = load_sjvn_logo()

# === Custom CSS with Clean Background - No Blur ===
background_style = ""
if sjvn_logo_base64:
    background_style = f"""
    .stApp {{
        background: url('data:image/png;base64,{sjvn_logo_base64}') center center / cover no-repeat fixed;
        background-size: cover;
    }}
    
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: -1;
    }}
    
    .main {{
        background: transparent !important;
    }}
    
    .main .block-container {{
        background: transparent !important;
        padding-top: 1rem !important;
    }}
    """
else:
    background_style = """
    .stApp {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    }
    
    .main {
        background: transparent !important;
    }
    
    .main .block-container {
        background: transparent !important;
        padding-top: 1rem !important;
    }
    """

st.markdown(f"""
<style>
    /* Background styling - Clean, no blur */
    {background_style}
    
    /* Header styling */
    .main-header {{
        background: rgba(30, 64, 175, 0.9);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    /* Chat container - Dynamic height, no initial scroll */
    .chat-container {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        min-height: 60vh;
        max-height: 70vh;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        position: relative;
    }}
    
    /* Chat messages styling */
    .stChatMessage {{
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 15px !important;
        margin: 0.5rem 0 !important;
        padding: 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    /* User message styling */
    .stChatMessage[data-testid="chat-message-user"] {{
        background: rgba(59, 130, 246, 0.4) !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        max-width: 80% !important;
        border-color: rgba(96, 165, 250, 0.4) !important;
    }}
    
    /* Assistant message styling */
    .stChatMessage[data-testid="chat-message-assistant"] {{
        background: rgba(255, 255, 255, 0.2) !important;
        margin-left: 0 !important;
        margin-right: auto !important;
        max-width: 80% !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }}
    
    /* Chat input styling */
    .stChatInput {{
        background: rgba(30, 64, 175, 0.4) !important;
        border-radius: 25px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    .stChatInput > div > div > input {{
        background: transparent !important;
        color: white !important;
        border: none !important;
        font-size: 1rem !important;
    }}
    
    .stChatInput > div > div > input::placeholder {{
        color: rgba(255, 255, 255, 0.7) !important;
    }}
    
    /* Sidebar styling - Clean, no blur */
    .css-1d391kg {{
        background: rgba(30, 64, 175, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
    }}
    
    .css-1d391kg .stMarkdown {{
        color: white !important;
    }}
    
    /* Sidebar elements */
    .sidebar-element {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }}
    
    /* Button styling */
    .stButton > button {{
        background: rgba(59, 130, 246, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    .stButton > button:hover {{
        background: rgba(59, 130, 246, 1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }}
    
    /* File uploader styling */
    .stFileUploader {{
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    .streamlit-expanderContent {{
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    /* Status indicators */
    .status-indicator {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        backdrop-filter: blur(10px);
    }}
    
    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning, .stError {{
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    /* Progress bar */
    .stProgress {{
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    /* Empty state styling */
    .empty-state {{
        text-align: center;
        padding: 3rem;
        color: rgba(255, 255, 255, 0.9);
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }}
    
    /* Text styling */
    .stMarkdown {{
        color: white !important;
    }}
    
    .stMarkdown p {{
        color: white !important;
    }}
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
        color: white !important;
    }}
    
    /* Links */
    a {{
        color: #93c5fd !important;
        text-decoration: none !important;
    }}
    
    a:hover {{
        color: #bfdbfe !important;
        text-decoration: underline !important;
    }}
    
    /* Columns */
    .stColumn {{
        background: transparent !important;
    }}
    
    /* Spinner */
    .stSpinner {{
        color: white !important;
    }}
    
    /* Scrollbar styling - Only appears when needed */
    .chat-container::-webkit-scrollbar {{
        width: 8px;
    }}
    
    .chat-container::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }}
    
    .chat-container::-webkit-scrollbar-thumb {{
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }}
    
    .chat-container::-webkit-scrollbar-thumb:hover {{
        background: rgba(255, 255, 255, 0.5);
    }}
    
    /* Non-chat pages container */
    .content-container {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }}
    
    /* Hide scrollbar initially if content fits */
    .chat-container {{
        scrollbar-width: thin;
        scrollbar-color: rgba(255, 255, 255, 0.3) transparent;
    }}
    
    /* Auto-hide scrollbar behavior */
    .chat-container:not(:hover)::-webkit-scrollbar {{
        width: 0px;
        background: transparent;
    }}
    
    .chat-container:hover::-webkit-scrollbar {{
        width: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# === Constants ===
PERSIST_DIR = "chroma_store"
SAVE_DIR = "uploaded_pdfs"
FILE_LOG = os.path.join(PERSIST_DIR, "uploaded_files.txt")
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# === Session State Initialization ===
if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# === Helper Functions ===
def create_download_link(file_path, file_name):
    """Create a download link for a file"""
    try:
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_name}"
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}">📥 Download {file_name}</a>'
        return href
    except Exception as e:
        return f"❌ Error creating download link for {file_name}: {str(e)}"

def load_existing_vectordb():
    """Load existing vector database if available"""
    try:
        if os.path.exists(os.path.join(PERSIST_DIR, "chroma-collections.parquet")):
            embedding = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
            return vectordb
    except Exception as e:
        st.error(f"Error loading existing database: {str(e)}")
    return None

@st.cache_resource
def load_chain(_vectordb):
    if _vectordb is None:
        return None
    retriever = _vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    llm = Ollama(
        model="mistral",
        temperature=0.1,
        top_p=0.9,
        num_ctx=2048
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

def save_chat_history(chat_id, history):
    """Save chat history to file"""
    try:
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
        with open(file_path, "w") as f:
            json.dump({
                "id": chat_id,
                "timestamp": datetime.now().isoformat(),
                "messages": history
            }, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

def load_chat_history(chat_id):
    """Load chat history from file"""
    try:
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                return data.get("messages", [])
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
    return []

def get_all_chat_sessions():
    """Get all chat sessions"""
    try:
        sessions = []
        for filename in os.listdir(CHAT_HISTORY_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(CHAT_HISTORY_DIR, filename)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": data.get("id"),
                        "timestamp": data.get("timestamp"),
                        "message_count": len(data.get("messages", []))
                    })
        return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)
    except Exception as e:
        st.error(f"Error loading chat sessions: {str(e)}")
        return []

def create_new_chat():
    """Create a new chat session"""
    chat_id = f"chat_{int(time.time())}"
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_history = []
    return chat_id

def load_documents(new_files):
    all_docs = []
    for uploaded_file in new_files:
        file_path = os.path.join(SAVE_DIR, uploaded_file.name)
        uploaded_file.seek(0)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    table_texts = []
                    
                    if tables:
                        for table in tables:
                            if table:
                                table_str = "\n".join([
                                    ", ".join(str(cell) if cell is not None else "" for cell in row)
                                    for row in table if row
                                ])
                                if table_str.strip():
                                    table_texts.append(table_str)

                    if table_texts:
                        combined_content = f"Page {i+1} Text:\n{text.strip()}\n\nExtracted Tables:\n" + "\n\n".join(table_texts)
                    else:
                        combined_content = f"Page {i+1} Text:\n{text.strip()}"
                    
                    if combined_content.strip():
                        doc = Document(page_content=combined_content, metadata={
                            "source": uploaded_file.name,
                            "path": file_path,
                            "page": i + 1
                        })
                        all_docs.append(doc)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(all_docs)

def create_or_update_vectordb(new_docs):
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    
    if os.path.exists(os.path.join(PERSIST_DIR, "chroma-collections.parquet")):
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
        st.info("🔁 Adding new documents to existing vector DB...")
    else:
        vectordb = None
        st.info("🆕 Creating a new vector DB from documents...")

    total = len(new_docs)
    progress_bar = st.progress(0, text="🔄 Starting embedding...")
    batch_size = 10
    start_time = time.time()

    try:
        if vectordb:
            for i in range(0, total, batch_size):
                batch = new_docs[i:i+batch_size]
                vectordb.add_documents(batch)
                
                elapsed = time.time() - start_time
                progress = min((i + len(batch)) / total, 1.0)
                avg_per_batch = elapsed / ((i // batch_size) + 1)
                remaining_batches = (total - i - len(batch)) // batch_size
                remaining_time = avg_per_batch * remaining_batches
                eta = time.strftime("%M:%S", time.gmtime(remaining_time))
                
                progress_bar.progress(
                    progress,
                    text=f"🔄 Processing batch {(i//batch_size)+1}/{(total//batch_size)+1} | ⏳ ETA: {eta}"
                )
        else:
            vectordb = Chroma.from_documents(
                documents=new_docs,
                embedding=embedding,
                persist_directory=PERSIST_DIR
            )
            progress_bar.progress(1.0, text="🔄 Creating new vector database...")

        progress_bar.empty()
        st.success("✅ Embedding complete and vector DB updated.")
        vectordb.persist()
        return vectordb
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error creating vector database: {str(e)}")
        return None

def check_new_files(files):
    if not os.path.exists(FILE_LOG):
        with open(FILE_LOG, "w") as f:
            pass
    try:
        with open(FILE_LOG, "r") as f:
            processed_files = set(line.strip() for line in f.readlines() if line.strip())
    except:
        processed_files = set()

    new_files = []
    already_exists = []
    for file in files:
        if file.name not in processed_files:
            new_files.append(file)
        else:
            already_exists.append(file.name)
    return new_files, already_exists

def update_file_log(files):
    try:
        with open(FILE_LOG, "a") as f:
            for file in files:
                f.write(file.name + "\n")
    except Exception as e:
        st.error(f"Error updating file log: {str(e)}")

# === Initialize Vector Database ===
if st.session_state.vectordb is None:
    st.session_state.vectordb = load_existing_vectordb()
    if st.session_state.vectordb:
        st.session_state.qa_chain = load_chain(st.session_state.vectordb)

# === Main Layout ===
# Header
st.markdown("""
<div class="main-header">
    <h1>💬 SJVNL - Suvidha Chatbot</h1>
    <p style="font-size: 1.2em; margin-top: 1rem; opacity: 0.9;">
        Intelligent PDF Document Assistant powered by Local AI
    </p>
</div>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
with st.sidebar:
    st.markdown('<div class="sidebar-element">', unsafe_allow_html=True)
    st.markdown("### 🧭 Navigation")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    pages = ["Chat", "Document Upload", "Chat History", "System Status", "Settings"]
    for page in pages:
        if st.button(f"📄 {page}" if page == "Chat" else f"📁 {page}" if page == "Document Upload" 
                    else f"📜 {page}" if page == "Chat History" else f"🔧 {page}" if page == "System Status"
                    else f"⚙ {page}"):
            st.session_state.current_page = page
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown('<div class="sidebar-element">', unsafe_allow_html=True)
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat"):
            create_new_chat()
            st.session_state.current_page = "Chat"
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat"):
            if st.session_state.current_chat_id and st.session_state.chat_history:
                save_chat_history(st.session_state.current_chat_id, st.session_state.chat_history)
                st.success("Chat saved!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Current Status
    st.markdown('<div class="sidebar-element">', unsafe_allow_html=True)
    st.markdown("### 📊 Current Status")
    
    # Database status
    db_status = "🟢 Ready" if st.session_state.vectordb else "🔴 No Database"
    st.markdown(f'<div class="status-indicator">*Database:* {db_status}</div>', unsafe_allow_html=True)
    
    # Chat status
    chat_status = f"🟢 Active ({len(st.session_state.chat_history)} messages)" if st.session_state.chat_history else "🔴 No Active Chat"
    st.markdown(f'<div class="status-indicator">*Chat:* {chat_status}</div>', unsafe_allow_html=True)
    
    # Document count
    doc_count = 0
    if os.path.exists(FILE_LOG):
        try:
            with open(FILE_LOG, "r") as f:
                doc_count = len([line.strip() for line in f.readlines() if line.strip()])
        except:
            pass
    st.markdown(f'<div class="status-indicator">*Documents:* {doc_count}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# === Main Content Area ===
if st.session_state.current_page == "Chat":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Initialize chat if needed
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = create_new_chat()
    
    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <h3>🤖 Welcome to SJVNL Chatbot!</h3>
            <p>Start a conversation by typing your question below.</p>
            <p>I can help you with information from uploaded documents or general queries.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    prompt = st.chat_input("Ask me anything...")
    
    if prompt:
        # Add user message
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                if st.session_state.qa_chain:
                    # Use document-based QA
                    result = st.session_state.qa_chain(prompt)
                    answer = result["result"]
                    source_docs = result["source_documents"]
                    
                    # Group results by file
                    file_to_pages = {}
                    file_to_path = {}
                    for doc in source_docs:
                        file = doc.metadata.get("source")
                        page = doc.metadata.get("page")
                        path = doc.metadata.get("path")
                        if file and page is not None:
                            file_to_pages.setdefault(file, set()).add(page)
                            file_to_path[file] = path
                    
                    if file_to_pages:
                        file_info_lines = []
                        for file, pages in file_to_pages.items():
                            page_list_str = ", ".join(f"Page {p}" for p in sorted(pages))
                            if file in file_to_path and os.path.exists(file_to_path[file]):
                                download_link = create_download_link(file_to_path[file], file)
                            else:
                                download_link = f"📄 {file} (file not accessible)"
                            file_info_lines.append(f"{download_link} — *{page_list_str}*")
                        
                        final_response = f"{answer}\n\n🔎 *Sources:\n\n" + "\n".join(file_info_lines) + "\n\n*Response generated by 🤖 Local Ollama (Mistral)"
                    else:
                        final_response = f"{answer}\n\n*Response generated by 🤖 Local Ollama (Mistral)*"
                else:
                    # Fallback to general LLM response
                    try:
                        llm = Ollama(model="mistral", temperature=0.1)
                        answer = llm(prompt)
                        final_response = f"{answer}\n\n*Response generated by 🤖 Local Ollama (Mistral)\n\n⚠ **Note:* No documents are loaded. Upload documents for document-specific queries."
                    except:
                        final_response = "❌ I'm sorry, but I'm currently unable to process your request. Please ensure Ollama is running and try again."
                
            except Exception as e:
                final_response = f"❌ Error generating response: {str(e)}"
        
        # Add assistant response
        st.chat_message("assistant").markdown(final_response, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
        
        # Auto-save chat
        if st.session_state.current_chat_id:
            save_chat_history(st.session_state.current_chat_id, st.session_state.chat_history)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "Document Upload":
    st.markdown("### 📤 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base"
    )
    
    if uploaded_files:
        new_files, duplicates = check_new_files(uploaded_files)
        
        if duplicates:
            st.warning(f"🚫 The following file(s) already exist:\n\n" + "\n".join(duplicates))
        
        if new_files:
            if st.button("📚 Process New Documents"):
                with st.spinner("📚 Processing documents..."):
                    docs = load_documents(new_files)
                    if docs:
                        vectordb = create_or_update_vectordb(docs)
                        if vectordb:
                            st.session_state.vectordb = vectordb
                            st.session_state.qa_chain = load_chain(vectordb)
                            update_file_log(new_files)
                            st.success("✅ Documents processed and added to database!")
                        else:
                            st.error("❌ Failed to process documents.")
                    else:
                        st.error("❌ No documents were processed successfully.")
    
    # Display current documents
    st.markdown("### 📋 Current Documents")
    if os.path.exists(FILE_LOG):
        try:
            with open(FILE_LOG, "r") as f:
                files = [line.strip() for line in f.readlines() if line.strip()]
                if files:
                    for i, file in enumerate(files, 1):
                        st.markdown(f"{i}.** {file}")
                else:
                    st.info("No documents uploaded yet.")
        except:
            st.info("No documents uploaded yet.")
    else:
        st.info("No documents uploaded yet.")

elif st.session_state.current_page == "Chat History":
    st.markdown("### 📜 Chat History")
    
    sessions = get_all_chat_sessions()
    
    if sessions:
        for session in sessions:
            with st.expander(f"Chat Session - {session['timestamp'][:19]} ({session['message_count']} messages)"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"*Session ID:* {session['id']}")
                    st.markdown(f"*Messages:* {session['message_count']}")
                    st.markdown(f"*Date:* {session['timestamp'][:19]}")
                with col2:
                    if st.button(f"Load", key=f"load_{session['id']}"):
                        st.session_state.current_chat_id = session['id']
                        st.session_state.chat_history = load_chat_history(session['id'])
                        st.session_state.current_page = "Chat"
                        st.rerun()
    else:
        st.info("No chat history available.")

elif st.session_state.current_page == "System Status":
    st.markdown("### 🔧 System Status")
    
    # Ollama status
    try:
        llm = Ollama(model="mistral")
        # Try a simple test
        test_response = llm("Hello")
        ollama_status = "🟢 Connected"
        ollama_details = "Mistral model is responsive"
    except:
        ollama_status = "🔴 Disconnected"
        ollama_details = "Cannot connect to Ollama"
    
    st.markdown(f"*Ollama Status:* {ollama_status}")
    st.markdown(f"*Details:* {ollama_details}")
    
    # Database status
    db_status = "🟢 Ready" if st.session_state.vectordb else "🔴 No Database"
    st.markdown(f"*Vector Database:* {db_status}")
    
    # File system status
    st.markdown("### 📁 File System")
    st.markdown(f"*Save Directory:* {SAVE_DIR}")
    st.markdown(f"*Persist Directory:* {PERSIST_DIR}")
    st.markdown(f"*Chat History Directory:* {CHAT_HISTORY_DIR}")
    
    # Performance metrics
    st.markdown("### 📊 Performance")
    if st.session_state.vectordb:
        try:
            # Get collection info
            st.markdown("*Database Statistics:*")
            st.markdown("- Vector database is loaded and ready")
            st.markdown("- Retrieval system is active")
        except:
            st.markdown("- Database information unavailable")
    else:
        st.markdown("- No vector database loaded")

elif st.session_state.current_page == "Settings":
    st.markdown("### ⚙ Settings")
    
    # Chat settings
    st.markdown("#### 💬 Chat Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear Current Chat"):
            st.session_state.chat_history = []
            st.session_state.current_chat_id = None
            st.success("Current chat cleared!")
    
    with col2:
        if st.button("🗂 Clear All Chat History"):
            try:
                for file in os.listdir(CHAT_HISTORY_DIR):
                    if file.endswith('.json'):
                        os.remove(os.path.join(CHAT_HISTORY_DIR, file))
                st.success("All chat history cleared!")
            except Exception as e:
                st.error(f"Error clearing chat history: {str(e)}")
    
    # Database settings
    st.markdown("#### 🗄 Database Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload Database"):
            st.session_state.vectordb = load_existing_vectordb()
            if st.session_state.vectordb:
                st.session_state.qa_chain = load_chain(st.session_state.vectordb)
                st.success("Database reloaded!")
            else:
                st.warning("No database found to reload.")
    
    with col2:
        if st.button("🗑 Clear Database"):
            if os.path.exists(PERSIST_DIR):
                import shutil
                shutil.rmtree(PERSIST_DIR)
                os.makedirs(PERSIST_DIR, exist_ok=True)
                if os.path.exists(FILE_LOG):
                    os.remove(FILE_LOG)
                st.session_state.vectordb = None
                st.session_state.qa_chain = None
                st.success("Database cleared!")
    
    # Model settings
    st.markdown("#### 🤖 Model Settings")
    
    st.markdown("*Current Configuration:*")
    st.markdown("- *LLM Model:* Mistral")
    st.markdown("- *Embedding Model:* nomic-embed-text")
    st.markdown("- *Temperature:* 0.1")
    st.markdown("- *Context Window:* 2048 tokens")
    st.markdown("- *Retrieval Documents:* 5")
    
    # Performance tips
    st.markdown("#### ⚡ Performance Tips")
    
    with st.expander("🔧 Optimization Guidelines"):
        st.markdown("""
        *To improve query speed:*
        - Use smaller PDF files when possible
        - Ask specific questions rather than broad ones
        - Ensure Ollama is running properly
        - Clear chat history periodically
        
        *Current optimizations:*
        - Batch processing for vector storage
        - Optimized retrieval parameters
        - Efficient table extraction
        - Local processing for privacy
        - Persistent vector database
        """)
    
    # About section
    st.markdown("#### ℹ About")
    
    with st.expander("📋 Application Information"):
        st.markdown("""
        *SJVNL Suvidha Chatbot*
        
        This is a local AI-powered document assistant that:
        - Processes PDF documents locally
        - Uses Ollama for inference
        - Maintains privacy by not sending data externally
        - Provides accurate answers with source references
        - Supports chat history and session management
        
        *Version:* 2.0
        *Model:* Mistral via Ollama
        *Framework:* Streamlit + LangChain
        """)
    
    # Export/Import settings
    st.markdown("#### 📤 Export/Import")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Export Chat History"):
            try:
                sessions = get_all_chat_sessions()
                if sessions:
                    export_data = {
                        "export_timestamp": datetime.now().isoformat(),
                        "sessions": sessions
                    }
                    export_json = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="💾 Download Chat History",
                        data=export_json,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No chat history to export.")
            except Exception as e:
                st.error(f"Error exporting chat history: {str(e)}")
    
    with col2:
        st.markdown("Import functionality coming soon")
