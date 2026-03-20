import os
from tools.rag_storage import MarketMindStorage

def run_ingestion():
    db_manager = MarketMindStorage()
    
    # Path to your documents folder
    docs_folder = "../data"
    
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        print(f"📁 Created '{docs_folder}' folder. Put your PDFs there and run again.")
        return

    files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]
    
    if not files:
        print("❌ No PDFs found in 'data_docs'.")
        return

    print(f"🚀 Found {len(files)} files. Starting ingestion...")
    
    for file in files:
        file_path = os.path.join(docs_folder, file)
        print(f"📄 Processing {file}...")
        db_manager.add_document(file_path)
    
    print("✅ RAG Database (ChromaDB) is now ready!")

if _name_ == "_main_":
    run_ingestion()