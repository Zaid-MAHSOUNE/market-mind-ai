import os
import streamlit as st
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings

# Keep your absolute path logic - it fixed the KeyError!
ABS_PATH = os.path.abspath(os.path.join(os.getcwd(), "data", "chroma_db"))

class MarketMindStorage:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        
        os.makedirs(os.path.dirname(ABS_PATH), exist_ok=True)
        
        # This now uses the modern langchain_chroma class
        self.vectorstore = Chroma(
            persist_directory=ABS_PATH,
            embedding_function=self.embeddings,
            collection_name="market_mind_collection"
        )
        print(f"✅ Modern ChromaDB initialized at: {ABS_PATH}")

    def add_document(self, file_path):
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        self.vectorstore.add_documents(chunks)
        # Note: 'persist()' is no longer needed in the new version; it's automatic.

    def search(self, query):
        """Fonction de recherche appelée par l'agent."""
        docs = self.vectorstore.similarity_search(query, k=4)
        context = "\n---\n".join([d.page_content for d in docs])
        return context
    


@st.cache_resource
def get_storage_engine():
        """Caches the heavy embedding model and DB connection."""
        return MarketMindStorage()