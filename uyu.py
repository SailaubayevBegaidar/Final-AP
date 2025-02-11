import streamlit as st
import hashlib
import sqlite3
import os
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import OllamaLLM
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.ollama import Ollama

# --- Constants and Initialization ---
llm_model = "llama3"
chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
collection_name = "knowledge_base"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize ChromaDB
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(name=collection_name, metadata={"description": "Knowledge base for RAG"})

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Functions from gfg.py ---
def generate_embeddings(documents):
    return embedding_model.encode(documents)

def save_embeddings(documents, ids):
    embeddings = generate_embeddings(documents)
    collection.add(documents=documents, ids=ids, embeddings=embeddings)

def query_chromadb(query_text, n_results=1):
    query_embedding = generate_embeddings([query_text])
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results["documents"][0] if results["documents"] else "No relevant documents found."

def google_search(query):
    API_KEY = "227e6513442cfaa35c62f17d43e8f0de50c3450af033c0ed8f9e7bee4de393c2"
    url = f"https://serpapi.com/search.json?q={query}&api_key={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get("organic_results", [])
            if results:
                return "\n".join(item.get("snippet", "") for item in results)
        return ""
    except Exception as e:
        st.error(f"Error searching Google: {e}")
        return ""

def wrap_text(text, max_length=80):
    return "\n".join([text[i:i + max_length] for i in range(0, len(text), max_length)])

def extract_text_from_file(file):
    if file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
        os.unlink(temp_file.name)
        return [doc.page_content for doc in documents]
    elif file.type == "text/plain":
        return [file.read().decode("utf-8")]
    else:
        st.error("Unsupported file type.")
        return []

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def process_query():
    initialize_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.text_input("Ask a question:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Searching knowledge base and web..."):
            # Сначала ищем в ChromaDB
            chroma_context = query_chromadb(user_input, n_results=3)
            
            # Если в ChromaDB ничего не найдено, ищем в Google
            if chroma_context == "No relevant documents found.":
                context = google_search(user_input)
                source = "web search"
            else:
                context = chroma_context
                source = "knowledge base"

            prompt = f"Context (from {source}):\n{context}\n\nQuestion:\n{user_input}\nAnswer:"

            try:
                llm = OllamaLLM(model=llm_model, base_url="http://localhost:11434")
                response = llm.invoke(prompt)

                st.session_state.messages.append({"role": "assistant", "content": response})

                with st.chat_message("assistant"):
                    st.write(response)

            except Exception as e:
                st.error(f"Error: {e}")

# --- Authentication Functions ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0] == make_hash(password)
    return False

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users VALUES (?,?)', (username, make_hash(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''

def main():
    init_db()
    init_session_state()

    st.set_page_config(page_title="Knowledge Navigator", layout="wide")

    if not st.session_state.logged_in:
        st.title("Welcome to Knowledge Navigator")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.header("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if check_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            st.header("Register")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            
            if st.button("Register"):
                if reg_password != reg_password_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif add_user(reg_username, reg_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

    else:
        st.title(f"Welcome back, {st.session_state.username}!")
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.rerun()

        # Document Management
        with st.expander("📚 Manage Documents"):
            st.subheader("➕ Add a New Document")
            uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
            new_doc = st.text_area("Alternatively, enter document text manually:")

            if uploaded_file:
                document_text = extract_text_from_file(uploaded_file)
                if document_text:
                    wrapped_text = [wrap_text(content) for content in document_text]
                    for i, content in enumerate(wrapped_text):
                        new_id = f"doc_{collection.count() + 1}_{i}"
                        save_embeddings([content], [new_id])
                    st.success("Document added successfully! Refresh to view the updated list.")
            elif new_doc.strip():
                new_id = f"doc_{collection.count() + 1}"
                save_embeddings([new_doc], [new_id])
                st.success("Document added successfully! Refresh to view the updated list.")
            elif st.button("Add Document") and not new_doc.strip():
                st.error("Document text cannot be empty!")

        # Chat Interface
        with st.expander("💬 Chat with the AI"):
            process_query()

if __name__ == "__main__":
    main()
