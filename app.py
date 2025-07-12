
# ####perfectly working code for single file upload

# import streamlit as st
# import pdfplumber
# from langchain.schema import Document
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OllamaEmbeddings
# from langchain.llms import Ollama
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# import tempfile
# import os
# import time

# # === Streamlit UI Setup ===
# st.set_page_config(page_title="📄 PDF Chatbot (Ollama)", layout="centered")
# st.title("💬 SJVNL - Suvidha Chatbot")

# # === Constants ===
# PERSIST_DIR = "chroma_store"
# SAVE_DIR = "uploaded_pdfs"
# FILE_LOG = os.path.join(PERSIST_DIR, "uploaded_files.txt")
# os.makedirs(SAVE_DIR, exist_ok=True)

# # === File Upload ===
# uploaded_files = st.file_uploader("📤 Upload PDF files", type=["pdf"], accept_multiple_files=True)

# # === Load Documents ===
# def load_documents(_new_files):
#     all_docs = []

#     for uploaded_file in _new_files:
#         # Save to disk for download later
#         file_path = os.path.join(SAVE_DIR, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())

#         with pdfplumber.open(file_path) as pdf:
#             for i, page in enumerate(pdf.pages):
#                 text = page.extract_text() or ""
#                 tables = page.extract_tables()

#                 table_texts = []
#                 for table in tables:
#                     table_str = "\n".join([
#                         ", ".join(str(cell) if cell is not None else "" for cell in row)
#                         for row in table if row
#                     ])
#                     table_texts.append(table_str)

#                 combined_content = f"Page {i+1} Text:\n{text.strip()}\n\nExtracted Tables:\n" + "\n\n".join(table_texts)
#                 doc = Document(page_content=combined_content, metadata={
#                     "source": uploaded_file.name,
#                     "path": file_path,
#                     "page": i + 1
#                 })
#                 all_docs.append(doc)

#     splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#     return splitter.split_documents(all_docs)

# # === Create or Update Vector Store ===
# def create_or_update_vectordb(new_docs):
#     embedding = OllamaEmbeddings(model="nomic-embed-text")

#     if os.path.exists(os.path.join(PERSIST_DIR, "chroma-collections.parquet")):
#         vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
#         st.info("🔁 Adding new documents to existing vector DB...")
#     else:
#         vectordb = None
#         st.info("🆕 Creating a new vector DB from documents...")

#     total = len(new_docs)
#     progress_bar = st.progress(0, text="🔄 Starting embedding...")
#     embedded_chunks = []

#     start_time = time.time()

#     for i, doc in enumerate(new_docs):
#         embedded = embedding.embed_query(doc.page_content)
#         embedded_chunks.append(doc)

#         elapsed = time.time() - start_time
#         avg_per_chunk = elapsed / (i + 1)
#         remaining = avg_per_chunk * (total - i - 1)
#         eta = time.strftime("%M:%S", time.gmtime(remaining))

#         progress_bar.progress(
#             (i + 1) / total,
#             text=f"🔄 Embedding chunk {i + 1}/{total} | ⏳ ETA: {eta}"
#         )

#     progress_bar.empty()
#     st.success("✅ Embedding complete and vector DB updated.")

#     if vectordb:
#         vectordb.add_documents(embedded_chunks)
#     else:
#         vectordb = Chroma.from_documents(
#             documents=embedded_chunks,
#             embedding=embedding,
#             persist_directory=PERSIST_DIR
#         )

#     vectordb.persist()
#     return vectordb

# # === Load LLM Chain ===
# @st.cache_resource
# def load_chain(_vectordb):
#     retriever = _vectordb.as_retriever()
#     llm = Ollama(model="mistral")
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True  # 🧠 critical for follow-up logic
#     )

# # === Check for Duplicate Files ===
# def check_new_files(_files):
#     if not os.path.exists(FILE_LOG):
#         open(FILE_LOG, "w").close()

#     with open(FILE_LOG, "r") as f:
#         processed_files = set(line.strip() for line in f.readlines())

#     new_files = []
#     already_exists = []
#     for file in _files:
#         if file.name not in processed_files:
#             new_files.append(file)
#         else:
#             already_exists.append(file.name)

#     return new_files, already_exists

# def update_file_log(_files):
#     with open(FILE_LOG, "a") as f:
#         for file in _files:
#             f.write(file.name + "\n")

# ## === Processing PDFs ===
# if uploaded_files:
#     new_files, duplicates = check_new_files(uploaded_files)

#     if duplicates:
#         st.warning(f"🚫 The following file(s) already exist in the database and were skipped:\n\n" + "\n".join(duplicates))

#     if new_files:
#         with st.spinner("📚 Reading and indexing new PDFs..."):
#             docs = load_documents(new_files)
#             vectordb = create_or_update_vectordb(docs)
#             update_file_log(new_files)
#             st.success("✅ New documents added to the database.")
#     else:
#         embedding = OllamaEmbeddings(model="nomic-embed-text")
#         vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

#     qa_chain = load_chain(vectordb)

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     prompt = st.chat_input("Ask a question based on the uploaded PDFs...")
#     if prompt:
#         st.chat_message("user").markdown(prompt)
#         st.session_state.chat_history.append({"role": "user", "content": prompt})

#         with st.spinner("Thinking..."):
#             result = qa_chain(prompt)
#             answer = result["result"]
#             source_docs = result["source_documents"]

#             # Group results by file
#             file_to_pages = {}
#             file_to_path = {}
#             for doc in source_docs:
#                 file = doc.metadata.get("source")
#                 page = doc.metadata.get("page")
#                 path = doc.metadata.get("path")
#                 if file and page is not None:
#                     file_to_pages.setdefault(file, set()).add(page)
#                     file_to_path[file] = path

#             if not file_to_pages:
#                 final_response = "⚠️ I couldn't find relevant content in the uploaded documents for your question."
#             else:
#                 file_info_lines = []
#                 for file, pages in file_to_pages.items():
#                     page_list_str = ", ".join(f"Page {p}" for p in sorted(pages))
#                     download_link = f"[📥 Download {file}](./{file_to_path[file]})"
#                     file_info_lines.append(f"{download_link} — **{page_list_str}**")

#                 final_response = f"{answer}\n\n🔎 **Relevant pages and files:**\n\n" + "\n".join(file_info_lines)

#         st.chat_message("assistant").markdown(final_response)
#         st.session_state.chat_history.append({"role": "assistant", "content": final_response})


# else:
#     st.info("Please upload at least one PDF file to start chatting.")







# # Claude AI code - local device 

# import streamlit as st
# import pdfplumber
# from langchain.schema import Document
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OllamaEmbeddings
# from langchain.llms import Ollama
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# import tempfile
# import os
# import time
# import base64

# # === Streamlit UI Setup ===
# st.set_page_config(page_title="📄 PDF Chatbot (Ollama)", layout="centered")
# st.title("💬 SJVNL - Suvidha Chatbot")

# # === Constants ===
# PERSIST_DIR = "chroma_store"
# SAVE_DIR = "uploaded_pdfs"
# FILE_LOG = os.path.join(PERSIST_DIR, "uploaded_files.txt")
# os.makedirs(SAVE_DIR, exist_ok=True)

# # === Helper function to create download link ===
# def create_download_link(file_path, file_name):
#     """Create a download link for a file"""
#     try:
#         with open(file_path, "rb") as f:
#             bytes_data = f.read()
#         b64 = base64.b64encode(bytes_data).decode()
#         href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}">📥 Download {file_name}</a>'
#         return href
#     except Exception as e:
#         return f"❌ Error creating download link for {file_name}"

# # === File Upload ===
# uploaded_files = st.file_uploader("📤 Upload PDF files", type=["pdf"], accept_multiple_files=True)

# # === Load Documents (Optimized) ===
# def load_documents(_new_files):
#     all_docs = []

#     for uploaded_file in _new_files:
#         # Save to disk for download later
#         file_path = os.path.join(SAVE_DIR, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())

#         with pdfplumber.open(file_path) as pdf:
#             for i, page in enumerate(pdf.pages):
#                 text = page.extract_text() or ""
                
#                 # Only extract tables if they exist (performance optimization)
#                 tables = page.extract_tables()
#                 table_texts = []
                
#                 if tables:  # Only process if tables exist
#                     for table in tables:
#                         if table:  # Check if table is not empty
#                             table_str = "\n".join([
#                                 ", ".join(str(cell) if cell is not None else "" for cell in row)
#                                 for row in table if row
#                             ])
#                             if table_str.strip():  # Only add non-empty tables
#                                 table_texts.append(table_str)

#                 # Combine content more efficiently
#                 if table_texts:
#                     combined_content = f"Page {i+1} Text:\n{text.strip()}\n\nExtracted Tables:\n" + "\n\n".join(table_texts)
#                 else:
#                     combined_content = f"Page {i+1} Text:\n{text.strip()}"
                
#                 # Skip empty pages
#                 if combined_content.strip():
#                     doc = Document(page_content=combined_content, metadata={
#                         "source": uploaded_file.name,
#                         "path": file_path,
#                         "page": i + 1
#                     })
#                     all_docs.append(doc)

#     # Optimized chunking parameters
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=3000,  # Reduced from 7500 for faster processing
#         chunk_overlap=200,  # Increased overlap for better context
#         length_function=len
#     )
#     return splitter.split_documents(all_docs)

# # === Create or Update Vector Store (Optimized) ===
# def create_or_update_vectordb(new_docs):
#     embedding = OllamaEmbeddings(model="nomic-embed-text")

#     if os.path.exists(os.path.join(PERSIST_DIR, "chroma-collections.parquet")):
#         vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
#         st.info("🔁 Adding new documents to existing vector DB...")
#     else:
#         vectordb = None
#         st.info("🆕 Creating a new vector DB from documents...")

#     total = len(new_docs)
#     progress_bar = st.progress(0, text="🔄 Starting embedding...")
    
#     # Batch processing for better performance
#     batch_size = 10  # Process in batches of 10
#     start_time = time.time()

#     if vectordb:
#         # Add documents in batches to existing vectordb
#         for i in range(0, total, batch_size):
#             batch = new_docs[i:i+batch_size]
#             vectordb.add_documents(batch)
            
#             elapsed = time.time() - start_time
#             progress = min((i + len(batch)) / total, 1.0)
#             avg_per_batch = elapsed / ((i // batch_size) + 1)
#             remaining_batches = (total - i - len(batch)) // batch_size
#             remaining_time = avg_per_batch * remaining_batches
#             eta = time.strftime("%M:%S", time.gmtime(remaining_time))
            
#             progress_bar.progress(
#                 progress,
#                 text=f"🔄 Processing batch {(i//batch_size)+1}/{(total//batch_size)+1} | ⏳ ETA: {eta}"
#             )
#     else:
#         # Create new vectordb with all documents at once (faster for new DB)
#         vectordb = Chroma.from_documents(
#             documents=new_docs,
#             embedding=embedding,
#             persist_directory=PERSIST_DIR
#         )
#         progress_bar.progress(1.0, text="🔄 Creating new vector database...")

#     progress_bar.empty()
#     st.success("✅ Embedding complete and vector DB updated.")
    
#     vectordb.persist()
#     return vectordb

# # === Load LLM Chain (Optimized) ===
# @st.cache_resource
# def load_chain(_vectordb):
#     retriever = _vectordb.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 3}  # Reduced from default to get faster results
#     )
#     llm = Ollama(
#         model="mistral",
#         temperature=0.1,  # Lower temperature for more focused responses
#         top_p=0.9,
#         num_ctx=2048  # Reduced context window for faster processing
#     )
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type="stuff"  # Most efficient chain type for small contexts
#     )

# # === Check for Duplicate Files ===
# def check_new_files(_files):
#     if not os.path.exists(FILE_LOG):
#         open(FILE_LOG, "w").close()

#     with open(FILE_LOG, "r") as f:
#         processed_files = set(line.strip() for line in f.readlines())

#     new_files = []
#     already_exists = []
#     for file in _files:
#         if file.name not in processed_files:
#             new_files.append(file)
#         else:
#             already_exists.append(file.name)

#     return new_files, already_exists

# def update_file_log(_files):
#     with open(FILE_LOG, "a") as f:
#         for file in _files:
#             f.write(file.name + "\n")

# ## === Processing PDFs ===
# if uploaded_files:
#     new_files, duplicates = check_new_files(uploaded_files)

#     if duplicates:
#         st.warning(f"🚫 The following file(s) already exist in the database and were skipped:\n\n" + "\n".join(duplicates))

#     if new_files:
#         with st.spinner("📚 Reading and indexing new PDFs..."):
#             docs = load_documents(new_files)
#             vectordb = create_or_update_vectordb(docs)
#             update_file_log(new_files)
#             st.success("✅ New documents added to the database.")
#     else:
#         embedding = OllamaEmbeddings(model="nomic-embed-text")
#         vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

#     qa_chain = load_chain(vectordb)

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)

#     prompt = st.chat_input("Ask a question based on the uploaded PDFs...")
#     if prompt:
#         st.chat_message("user").markdown(prompt)
#         st.session_state.chat_history.append({"role": "user", "content": prompt})

#         with st.spinner("Thinking..."):
#             result = qa_chain(prompt)
#             answer = result["result"]
#             source_docs = result["source_documents"]

#             # Group results by file
#             file_to_pages = {}
#             file_to_path = {}
#             for doc in source_docs:
#                 file = doc.metadata.get("source")
#                 page = doc.metadata.get("page")
#                 path = doc.metadata.get("path")
#                 if file and page is not None:
#                     file_to_pages.setdefault(file, set()).add(page)
#                     file_to_path[file] = path

#             if not file_to_pages:
#                 final_response = "⚠️ I couldn't find relevant content in the uploaded documents for your question."
#             else:
#                 file_info_lines = []
#                 for file, pages in file_to_pages.items():
#                     page_list_str = ", ".join(f"Page {p}" for p in sorted(pages))
#                     # Create proper download link using base64 encoding
#                     download_link = create_download_link(file_to_path[file], file)
#                     file_info_lines.append(f"{download_link} — **{page_list_str}**")

#                 final_response = f"{answer}\n\n🔎 **Relevant pages and files:**\n\n" + "\n".join(file_info_lines)

#         st.chat_message("assistant").markdown(final_response, unsafe_allow_html=True)
#         st.session_state.chat_history.append({"role": "assistant", "content": final_response})

# else:
#     st.info("Please upload at least one PDF file to start chatting.")

# # === Performance Tips Section ===
# with st.expander("⚡ Performance Tips"):
#     st.markdown("""
#     **To improve query speed:**
#     - Use smaller PDF files when possible
#     - Ask specific questions rather than broad ones
#     - The system learns from your documents, so repeated similar queries will be faster
#     - Consider using a more powerful local model if available
    
#     **Current optimizations:**
#     - Reduced chunk size for faster embedding
#     - Batch processing for vector storage
#     - Optimized retrieval parameters
#     - Efficient table extraction
#     """)















# enhanced interface (1)

# import streamlit as st
# import pdfplumber
# from langchain.schema import Document
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OllamaEmbeddings
# from langchain.llms import Ollama
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# import tempfile
# import os
# import time
# import base64   

# # === Streamlit UI Setup ===
# st.set_page_config(
#     page_title="📄 SJVNL Asuvidha Chatbot", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # === Custom CSS for Navy Blue Theme ===
# st.markdown("""
# <style>
#     /* Main background */
#     .main {
#         background-color: #1e3a8a;
#         color: white;
#     }
    
#     /* Sidebar styling */
#     .css-1d391kg {
#         background-color: #1e40af;
#     }
    
#     /* Chat input styling */
#     .stChatInput > div > div > input {
#         background-color: #3b82f6;
#         color: white;
#         border: 1px solid #60a5fa;
#     }
    
#     /* File uploader styling */
#     .stFileUploader > div > div > div {
#         background-color: #3b82f6;
#         border: 2px dashed #60a5fa;
#     }
    
#     /* Progress bar styling */
#     .stProgress > div > div {
#         background-color: #60a5fa;
#     }
    
#     /* Header styling */
#     .main-header {
#         background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         margin-bottom: 2rem;
#         text-align: center;
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
#     }
    
#     /* Image container styling */
#     .image-container {
#         background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         margin-bottom: 2rem;
#         text-align: center;
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
#         min-height: 400px;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         flex-direction: column;
#     }
    
#     /* Chat container styling */
#     .chat-container {
#         background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         margin-bottom: 2rem;
#         box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
#         min-height: 600px;
#     }
    
#     /* Feature card styling */
#     .feature-card {
#         background: rgba(59, 130, 246, 0.3);
#         padding: 1.5rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#         border: 1px solid rgba(96, 165, 250, 0.5);
#         backdrop-filter: blur(10px);
#     }
    
#     /* Logo styling */
#     .logo-container {
#         background: rgba(255, 255, 255, 0.1);
#         padding: 3rem;
#         border-radius: 50%;
#         margin-bottom: 2rem;
#         backdrop-filter: blur(10px);
#     }
    
#     /* Stats card styling */
#     .stats-card {
#         background: rgba(59, 130, 246, 0.2);
#         padding: 1rem;
#         border-radius: 8px;
#         text-align: center;
#         border: 1px solid rgba(96, 165, 250, 0.3);
#         margin: 0.5rem 0;
#     }
    
#     /* Button styling */
#     .stButton > button {
#         background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
#         color: white;
#         border: none;
#         border-radius: 8px;
#         padding: 0.5rem 1rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
    
#     .stButton > button:hover {
#         background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%);
#         transform: translateY(-2px);
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
#     }
    
#     /* Expander styling */
#     .streamlit-expanderHeader {
#         background-color: rgba(59, 130, 246, 0.3);
#         border-radius: 8px;
#         color: white;
#     }
    
#     /* Success/Info/Warning message styling */
#     .stSuccess, .stInfo, .stWarning {
#         background-color: rgba(59, 130, 246, 0.2);
#         border-radius: 8px;
#         backdrop-filter: blur(10px);
#     }
    
#     /* Download link styling */
#     a {
#         color: #93c5fd;
#         text-decoration: none;
#         font-weight: 500;
#     }
    
#     a:hover {
#         color: #dbeafe;
#         text-decoration: underline;
#     }
# </style>
# """, unsafe_allow_html=True)

# # === Constants ===
# PERSIST_DIR = "chroma_store"
# SAVE_DIR = "uploaded_pdfs"
# FILE_LOG = os.path.join(PERSIST_DIR, "uploaded_files.txt")
# os.makedirs(SAVE_DIR, exist_ok=True)
# os.makedirs(PERSIST_DIR, exist_ok=True)

# # === Main Layout ===
# # Header
# st.markdown("""
# <div class="main-header">
#     <h1>💬 SJVNL - Suvidha Chatbot</h1>
#     <p style="font-size: 1.2em; margin-top: 1rem; opacity: 0.9;">
#         Intelligent PDF Document Assistant powered by Local AI
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # Create two columns for layout
# col1, col2 = st.columns([1, 2])

# # Left column - Image and Info Section
# with col1:
#     st.markdown("""
#     <div class="image-container">
#         <div class="logo-container">
#             <h2 style="margin: 0; font-size: 3em;">🤖</h2>
#         </div>
#         <h3 style="margin-bottom: 1rem; color: #dbeafe;">AI-Powered Assistant</h3>
#         <p style="text-align: center; opacity: 0.9;">
#             Upload your PDF documents and ask questions. Our intelligent system will analyze 
#             your documents and provide accurate answers with source references.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Features Section
#     st.markdown("""
#     <div class="feature-card">
#         <h4>🚀 Key Features</h4>
#         <ul style="margin: 0; padding-left: 1.5rem;">
#             <li>📄 Multi-PDF Support</li>
#             <li>🔍 Intelligent Search</li>
#             <li>📊 Table Extraction</li>
#             <li>💾 Local Processing</li>
#             <li>🔗 Source References</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Stats Section
#     if "chat_history" in st.session_state:
#         messages_count = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
#     else:
#         messages_count = 0
    
#     uploaded_count = 0
#     if os.path.exists(FILE_LOG):
#         try:
#             with open(FILE_LOG, "r") as f:
#                 uploaded_count = len([line.strip() for line in f.readlines() if line.strip()])
#         except:
#             uploaded_count = 0
    
#     st.markdown(f"""
#     <div class="stats-card">
#         <h4>📊 Session Stats</h4>
#         <p><strong>Documents Uploaded:</strong> {uploaded_count}</p>
#         <p><strong>Questions Asked:</strong> {messages_count}</p>
#     </div>
#     """, unsafe_allow_html=True)

# # Right column - Chat Interface
# with col2:
#     st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
#     # File Upload Section
#     st.markdown("### 📤 Upload Documents")
#     uploaded_files = st.file_uploader(
#         "Choose PDF files", 
#         type=["pdf"], 
#         accept_multiple_files=True,
#         help="Upload one or more PDF files to start chatting"
#     )
    
#     # === Helper Functions ===
#     def create_download_link(file_path, file_name):
#         """Create a download link for a file"""
#         try:
#             if not os.path.exists(file_path):
#                 return f"❌ File not found: {file_name}"
#             with open(file_path, "rb") as f:
#                 bytes_data = f.read()
#             b64 = base64.b64encode(bytes_data).decode()
#             href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}">📥 Download {file_name}</a>'
#             return href
#         except Exception as e:
#             return f"❌ Error creating download link for {file_name}: {str(e)}"

#     def load_documents(_new_files):
#         all_docs = []
#         for uploaded_file in _new_files:
#             file_path = os.path.join(SAVE_DIR, uploaded_file.name)
#             uploaded_file.seek(0)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.read())

#             try:
#                 with pdfplumber.open(file_path) as pdf:
#                     for i, page in enumerate(pdf.pages):
#                         text = page.extract_text() or ""
#                         tables = page.extract_tables()
#                         table_texts = []
                        
#                         if tables:
#                             for table in tables:
#                                 if table:
#                                     table_str = "\n".join([
#                                         ", ".join(str(cell) if cell is not None else "" for cell in row)
#                                         for row in table if row
#                                     ])
#                                     if table_str.strip():
#                                         table_texts.append(table_str)

#                         if table_texts:
#                             combined_content = f"Page {i+1} Text:\n{text.strip()}\n\nExtracted Tables:\n" + "\n\n".join(table_texts)
#                         else:
#                             combined_content = f"Page {i+1} Text:\n{text.strip()}"
                        
#                         if combined_content.strip():
#                             doc = Document(page_content=combined_content, metadata={
#                                 "source": uploaded_file.name,
#                                 "path": file_path,
#                                 "page": i + 1
#                             })
#                             all_docs.append(doc)
#             except Exception as e:
#                 st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#                 continue

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=3000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         return splitter.split_documents(all_docs)

#     def create_or_update_vectordb(new_docs):
#         embedding = OllamaEmbeddings(model="nomic-embed-text")
        
#         if os.path.exists(os.path.join(PERSIST_DIR, "chroma-collections.parquet")):
#             vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
#             st.info("🔁 Adding new documents to existing vector DB...")
#         else:
#             vectordb = None
#             st.info("🆕 Creating a new vector DB from documents...")

#         total = len(new_docs)
#         progress_bar = st.progress(0, text="🔄 Starting embedding...")
#         batch_size = 10
#         start_time = time.time()

#         try:
#             if vectordb:
#                 for i in range(0, total, batch_size):
#                     batch = new_docs[i:i+batch_size]
#                     vectordb.add_documents(batch)
                    
#                     elapsed = time.time() - start_time
#                     progress = min((i + len(batch)) / total, 1.0)
#                     avg_per_batch = elapsed / ((i // batch_size) + 1)
#                     remaining_batches = (total - i - len(batch)) // batch_size
#                     remaining_time = avg_per_batch * remaining_batches
#                     eta = time.strftime("%M:%S", time.gmtime(remaining_time))
                    
#                     progress_bar.progress(
#                         progress,
#                         text=f"🔄 Processing batch {(i//batch_size)+1}/{(total//batch_size)+1} | ⏳ ETA: {eta}"
#                     )
#             else:
#                 vectordb = Chroma.from_documents(
#                     documents=new_docs,
#                     embedding=embedding,
#                     persist_directory=PERSIST_DIR
#                 )
#                 progress_bar.progress(1.0, text="🔄 Creating new vector database...")

#             progress_bar.empty()
#             st.success("✅ Embedding complete and vector DB updated.")
#             vectordb.persist()
#             return vectordb
#         except Exception as e:
#             progress_bar.empty()
#             st.error(f"Error creating vector database: {str(e)}")
#             return None

#     @st.cache_resource
#     def load_chain(_vectordb):
#         if _vectordb is None:
#             return None
#         retriever = _vectordb.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 5}
#         )
#         llm = Ollama(
#             model="mistral",
#             temperature=0.1,
#             top_p=0.9,
#             num_ctx=2048
#         )
#         return RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             return_source_documents=True,
#             chain_type="stuff"
#         )

#     def check_new_files(_files):
#         if not os.path.exists(FILE_LOG):
#             with open(FILE_LOG, "w") as f:
#                 pass
#         try:
#             with open(FILE_LOG, "r") as f:
#                 processed_files = set(line.strip() for line in f.readlines() if line.strip())
#         except:
#             processed_files = set()

#         new_files = []
#         already_exists = []
#         for file in _files:
#             if file.name not in processed_files:
#                 new_files.append(file)
#             else:
#                 already_exists.append(file.name)
#         return new_files, already_exists

#     def update_file_log(_files):
#         try:
#             with open(FILE_LOG, "a") as f:
#                 for file in _files:
#                     f.write(file.name + "\n")
#         except Exception as e:
#             st.error(f"Error updating file log: {str(e)}")

#     # === Main Processing Logic ===
#     if uploaded_files:
#         new_files, duplicates = check_new_files(uploaded_files)

#         if duplicates:
#             st.warning(f"🚫 The following file(s) already exist in the database and were skipped:\n\n" + "\n".join(duplicates))

#         if new_files:
#             with st.spinner("📚 Reading and indexing new PDFs..."):
#                 docs = load_documents(new_files)
#                 if docs:
#                     vectordb = create_or_update_vectordb(docs)
#                     if vectordb:
#                         update_file_log(new_files)
#                         st.success("✅ New documents added to the database.")
#                     else:
#                         st.error("❌ Failed to create vector database.")
#                 else:
#                     st.error("❌ No documents were processed successfully.")
#         else:
#             try:
#                 embedding = OllamaEmbeddings(model="nomic-embed-text")
#                 vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
#             except Exception as e:
#                 st.error(f"Error loading existing database: {str(e)}")
#                 vectordb = None

#         if vectordb:
#             qa_chain = load_chain(vectordb)
            
#             if qa_chain:
#                 st.markdown("### 💬 Chat Interface")
                
#                 if "chat_history" not in st.session_state:
#                     st.session_state.chat_history = []

#                 # Display chat history
#                 for msg in st.session_state.chat_history:
#                     with st.chat_message(msg["role"]):
#                         st.markdown(msg["content"], unsafe_allow_html=True)

#                 # Chat input
#                 prompt = st.chat_input("Ask a question based on the uploaded PDFs...")
#                 if prompt:
#                     st.chat_message("user").markdown(prompt)
#                     st.session_state.chat_history.append({"role": "user", "content": prompt})

#                     with st.spinner("🤔 Thinking..."):
#                         try:
#                             result = qa_chain(prompt)
#                             answer = result["result"]
#                             source_docs = result["source_documents"]

#                             # Group results by file
#                             file_to_pages = {}
#                             file_to_path = {}
#                             for doc in source_docs:
#                                 file = doc.metadata.get("source")
#                                 page = doc.metadata.get("page")
#                                 path = doc.metadata.get("path")
#                                 if file and page is not None:
#                                     file_to_pages.setdefault(file, set()).add(page)
#                                     file_to_path[file] = path

#                             if not file_to_pages:
#                                 final_response = "⚠️ I couldn't find relevant content in the uploaded documents for your question."
#                             else:
#                                 file_info_lines = []
#                                 for file, pages in file_to_pages.items():
#                                     page_list_str = ", ".join(f"Page {p}" for p in sorted(pages))
#                                     if file in file_to_path and os.path.exists(file_to_path[file]):
#                                         download_link = create_download_link(file_to_path[file], file)
#                                     else:
#                                         download_link = f"📄 {file} (file not accessible)"
#                                     file_info_lines.append(f"{download_link} — **{page_list_str}**")

#                                 final_response = f"{answer}\n\n🔎 **Relevant pages and files:**\n\n" + "\n".join(file_info_lines) + "\n\n*Response generated by 🤖 Local Ollama (Mistral)*"

#                         except Exception as e:
#                             final_response = f"❌ Error generating response: {str(e)}"

#                     st.chat_message("assistant").markdown(final_response, unsafe_allow_html=True)
#                     st.session_state.chat_history.append({"role": "assistant", "content": final_response})
#             else:
#                 st.error("❌ Failed to load the QA chain.")
#         else:
#             st.error("❌ Failed to load vector database.")
#     else:
#         st.info("👆 Please upload at least one PDF file to start chatting.")
#         st.markdown("""
#         <div style="text-align: center; margin-top: 2rem; opacity: 0.7;">
#             <p>Upload your PDF documents above and start asking questions!</p>
#             <p>The AI will analyze your documents and provide accurate answers with source references.</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# # === Sidebar with Additional Options ===
# with st.sidebar:
#     st.markdown("### 🔧 System Status")
    
#     # System status indicators
#     ollama_status = "🟢 Connected" if True else "🔴 Disconnected"  # You can add actual ollama connection check
#     st.markdown(f"**Ollama Status:** {ollama_status}")
#     st.markdown(f"**Model:** Mistral")
#     st.markdown(f"**Embedding Model:** nomic-embed-text")
    
#     st.markdown("---")
    
#     # Settings section
#     st.markdown("### ⚙️ Settings")
    
#     # Clear chat history button
#     if st.button("🗑️ Clear Chat History"):
#         if "chat_history" in st.session_state:
#             st.session_state.chat_history = []
#             st.success("Chat history cleared!")
    
#     # Clear database button
#     if st.button("🗂️ Clear Database"):
#         if os.path.exists(PERSIST_DIR):
#             import shutil
#             shutil.rmtree(PERSIST_DIR)
#             os.makedirs(PERSIST_DIR, exist_ok=True)
#             if os.path.exists(FILE_LOG):
#                 os.remove(FILE_LOG)
#             st.success("Database cleared!")
#             st.experimental_rerun()
    
#     st.markdown("---")
    
#     # Performance Tips
#     with st.expander("⚡ Performance Tips"):
#         st.markdown("""
#         **To improve query speed:**
#         - Use smaller PDF files when possible
#         - Ask specific questions rather than broad ones
#         - The system learns from your documents
#         - Ensure Ollama is running properly
        
#         **Current optimizations:**
#         - Batch processing for vector storage
#         - Optimized retrieval parameters
#         - Efficient table extraction
#         - Local processing for privacy
#         """)
    
#     # About section
#     with st.expander("ℹ️ About"):
#         st.markdown("""
#         **SJVNL Suvidha Chatbot**
        
#         This is a local AI-powered document assistant that:
#         - Processes PDF documents locally
#         - Uses Ollama for inference
#         - Maintains privacy by not sending data externally
#         - Provides accurate answers with source references
        
#         **Version:** 2.0
#         **Model:** Mistral via Ollama
#         """)




























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

# === Streamlit UI Setup ===
st.set_page_config(
    page_title="📄 SJVNL Asuvidha Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for Navy Blue Theme ===
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #1e3a8a;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e40af;
    }
    
    /* Navigation tabs styling */
    .nav-tab {
        background: rgba(59, 130, 246, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid rgba(96, 165, 250, 0.5);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-tab:hover {
        background: rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    .nav-tab.active {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        border-color: #93c5fd;
    }
    
    /* Chat input styling */
    .stChatInput > div > div > input {
        background-color: #3b82f6;
        color: white;
        border: 1px solid #60a5fa;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div {
        background-color: #3b82f6;
        border: 2px dashed #60a5fa;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #60a5fa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Chat container styling */
    .chat-container {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        min-height: 70vh;
    }
    
    /* Feature card styling */
    .feature-card {
        background: rgba(59, 130, 246, 0.3);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(96, 165, 250, 0.5);
        backdrop-filter: blur(10px);
    }
    
    /* Stats card styling */
    .stats-card {
        background: rgba(59, 130, 246, 0.2);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(96, 165, 250, 0.3);
        margin: 0.5rem 0;
    }
    
    /* History item styling */
    .history-item {
        background: rgba(59, 130, 246, 0.2);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid rgba(96, 165, 250, 0.3);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        color: white;
    }
    
    /* Success/Info/Warning message styling */
    .stSuccess, .stInfo, .stWarning {
        background-color: rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    /* Download link styling */
    a {
        color: #93c5fd;
        text-decoration: none;
        font-weight: 500;
    }
    
    a:hover {
        color: #dbeafe;
        text-decoration: underline;
    }
    
    /* Empty state styling */
    .empty-state {
        text-align: center;
        padding: 3rem;
        opacity: 0.7;
    }
    
    /* Chat message styling */
    .chat-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: rgba(96, 165, 250, 0.3);
        border-left: 4px solid #60a5fa;
    }
    
    .assistant-message {
        background: rgba(59, 130, 246, 0.2);
        border-left: 4px solid #3b82f6;
    }
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
    st.markdown("### 🧭 Navigation")
    
    # Navigation buttons
    pages = ["Chat", "Document Upload", "Chat History", "System Status", "Settings"]
    for page in pages:
        if st.button(f"📄 {page}" if page == "Chat" else f"📁 {page}" if page == "Document Upload" 
                    else f"📜 {page}" if page == "Chat History" else f"🔧 {page}" if page == "System Status"
                    else f"⚙️ {page}"):
            st.session_state.current_page = page
    
    st.markdown("---")
    
    # Quick Actions
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
    
    st.markdown("---")
    
    # Current Status
    st.markdown("### 📊 Current Status")
    
    # Database status
    db_status = "🟢 Ready" if st.session_state.vectordb else "🔴 No Database"
    st.markdown(f"**Database:** {db_status}")
    
    # Chat status
    chat_status = f"🟢 Active ({len(st.session_state.chat_history)} messages)" if st.session_state.chat_history else "🔴 No Active Chat"
    st.markdown(f"**Chat:** {chat_status}")
    
    # Document count
    doc_count = 0
    if os.path.exists(FILE_LOG):
        try:
            with open(FILE_LOG, "r") as f:
                doc_count = len([line.strip() for line in f.readlines() if line.strip()])
        except:
            pass
    st.markdown(f"**Documents:** {doc_count}")

# === Main Content Area ===
if st.session_state.current_page == "Chat":
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat Interface
    st.markdown("### 💬 Chat Interface")
    
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
                            file_info_lines.append(f"{download_link} — **{page_list_str}**")
                        
                        final_response = f"{answer}\n\n🔎 **Sources:**\n\n" + "\n".join(file_info_lines) + "\n\n*Response generated by 🤖 Local Ollama (Mistral)*"
                    else:
                        final_response = f"{answer}\n\n*Response generated by 🤖 Local Ollama (Mistral)*"
                else:
                    # Fallback to general LLM response
                    try:
                        llm = Ollama(model="mistral", temperature=0.1)
                        answer = llm(prompt)
                        final_response = f"{answer}\n\n*Response generated by 🤖 Local Ollama (Mistral)*\n\n⚠️ **Note:** No documents are loaded. Upload documents for document-specific queries."
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
                        st.markdown(f"**{i}.** {file}")
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
                    st.markdown(f"**Session ID:** {session['id']}")
                    st.markdown(f"**Messages:** {session['message_count']}")
                    st.markdown(f"**Date:** {session['timestamp'][:19]}")
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
    
    st.markdown(f"**Ollama Status:** {ollama_status}")
    st.markdown(f"**Details:** {ollama_details}")
    
    # Database status
    db_status = "🟢 Ready" if st.session_state.vectordb else "🔴 No Database"
    st.markdown(f"**Vector Database:** {db_status}")
    
    # File system status
    st.markdown("### 📁 File System")
    st.markdown(f"**Save Directory:** {SAVE_DIR}")
    st.markdown(f"**Persist Directory:** {PERSIST_DIR}")
    st.markdown(f"**Chat History Directory:** {CHAT_HISTORY_DIR}")
    
    # Performance metrics
    st.markdown("### 📊 Performance")
    if st.session_state.vectordb:
        try:
            # Get collection info
            st.markdown("**Database Statistics:**")
            st.markdown("- Vector database is loaded and ready")
            st.markdown("- Retrieval system is active")
        except:
            st.markdown("- Database information unavailable")
    else:
        st.markdown("- No vector database loaded")

elif st.session_state.current_page == "Settings":
    st.markdown("### ⚙️ Settings")
    
    # Chat settings
    st.markdown("#### 💬 Chat Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Current Chat"):
            st.session_state.chat_history = []
            st.session_state.current_chat_id = None
            st.success("Current chat cleared!")
    
    with col2:
        if st.button("🗂️ Clear All Chat History"):
            try:
                for file in os.listdir(CHAT_HISTORY_DIR):
                    if file.endswith('.json'):
                        os.remove(os.path.join(CHAT_HISTORY_DIR, file))
                st.success("All chat history cleared!")
            except Exception as e:
                st.error(f"Error clearing chat history: {str(e)}")
    
    # Database settings
    st.markdown("#### 🗄️ Database Settings")
    
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
        if st.button("🗑️ Clear Database"):
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
    
    st.markdown("**Current Configuration:**")
    st.markdown("- **LLM Model:** Mistral")
    st.markdown("- **Embedding Model:** nomic-embed-text")
    st.markdown("- **Temperature:** 0.1")
    st.markdown("- **Context Window:** 2048 tokens")
    st.markdown("- **Retrieval Documents:** 5")
    
    # Performance tips
    st.markdown("#### ⚡ Performance Tips")
    
    with st.expander("🔧 Optimization Guidelines"):
        st.markdown("""
        **To improve query speed:**
        - Use smaller PDF files when possible
        - Ask specific questions rather than broad ones
        - Ensure Ollama is running properly
        - Clear chat history periodically
        
        **Current optimizations:**
        - Batch processing for vector storage
        - Optimized retrieval parameters
        - Efficient table extraction
        - Local processing for privacy
        - Persistent vector database
        """)
    
    # About section
    st.markdown("#### ℹ️ About")
    
    with st.expander("📋 Application Information"):
        st.markdown("""
        **SJVNL Suvidha Chatbot**
        
        This is a local AI-powered document assistant that:
        - Processes PDF documents locally
        - Uses Ollama for inference
        - Maintains privacy by not sending data externally
        - Provides accurate answers with source references
        - Supports chat history and session management
        
        **Version:** 2.0
        **Model:** Mistral via Ollama
        **Framework:** Streamlit + LangChain
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
        st.markdown("*Import functionality coming soon*")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 1rem;">
    <p>🤖 SJVNL Suvidha Chatbot v2.0 | Powered by Ollama & LangChain</p>
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)