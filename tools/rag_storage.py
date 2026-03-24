import os
import streamlit as st
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings

# Keep your absolute path logic - it fixed the KeyError last time!
ABS_PATH = os.path.abspath(os.path.join(os.getcwd(), "data", "chroma_db"))

class MarketMindStorage:
    def __init__(self):
        # Initialize OpenAI embeddings to transform text into math vectors
        self.embeddings = OpenAIEmbeddings()
        
        # Ensure the parent directory exists, otherwise create it
        os.makedirs(os.path.dirname(ABS_PATH), exist_ok=True)
        
        # Using the modern langchain_chroma class for better stability
        # Link the store to the persistent directory and our embedding function
        self.vectorstore = Chroma(
            persist_directory=ABS_PATH,
            embedding_function=self.embeddings,
            collection_name="market_mind_collection"
        )
        print(f"✅ Modern ChromaDB initialized at: {ABS_PATH}")

    def add_document(self, file_path):
        """Splits a PDF into chunks and injects them into the database."""
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Loading the PDF using PyMuPDF (fast and reliable)
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # Split text: chunks of 800 characters with a 100-character overlap
        # This overlap prevents losing context between two blocks of text.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Add chunks to the vector store.
        # Note: 'persist()' is no longer needed in this version; it's automatic.
        self.vectorstore.add_documents(chunks)

    def search(self, query):
        """Search function called by the agent for the RAG process."""
        # Search for the top 4 text chunks most similar to the user query
        docs = self.vectorstore.similarity_search(query, k=4)
        
        # Concatenate everything cleanly so the AI can read it all at once
        context = "\n---\n".join([d.page_content for d in docs])
        return context

@st.cache_resource
def get_storage_engine():
    """
    Streamlit trick: Caches the heavy embedding model and DB connection
    to avoid reloading everything on every user interaction/re-run.
    """
    return MarketMindStorage()