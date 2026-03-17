import os
from dotenv import load_dotenv
from langchain_core.tools import Tool  
from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Configuration du dossier de stockage
CHROMA_PATH = "chroma_db"

class MarketMindStorage:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        # Initialisation de la base de données persistante
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=self.embeddings
        )

    def add_document(self, file_path):
        """Charge un PDF, le découpe et l'ajoute à la base Chroma existante."""
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Ajout et sauvegarde
        self.vectorstore.add_documents(chunks)
        print(f"✅ {len(chunks)} segments ajoutés à la base de données.")

    def get_tool(self):
        """Retourne l'outil utilisable par un Agent (ReAct)."""
        return Tool(
            name="Market_Knowledge_Base",
            func=self.search,
            description="Utile pour obtenir des informations précises sur des rapports financiers ou des documents techniques stockés."
        )

    def search(self, query):
        """Fonction de recherche appelée par l'agent."""
        docs = self.vectorstore.similarity_search(query, k=4)
        context = "\n---\n".join([d.page_content for d in docs])
        return context

# --- Exemple d'initialisation pour tes agents ---
# db_manager = MarketMindStorage()
# # db_manager.add_document("rapport_annuel.pdf") # À faire une seule fois
# rag_tool = db_manager.get_tool()