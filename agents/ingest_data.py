import os
from tools.rag_storage import MarketMindStorage

def run_ingestion():
    # Initialize the database manager (ChromaDB via our custom class)
    db_manager = MarketMindStorage()
    
    # This is where we store the PDFs to be processed
    docs_folder = "../data"
    
    # Quick safety check: if the folder doesn't exist, we create it
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        print(f"📁 Folder '{docs_folder}' created. Drop your PDFs in there and run the script again.")
        return

    # Grab only the files that end with .pdf
    files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]
    
    # If the folder is empty, we stop here—no point in continuing
    if not files:
        print("❌ No PDFs found in 'data_docs'.")
        return

    # Let's get to work!
    print(f"🚀 Found {len(files)} files. Starting ingestion...")
    
    # Loop through each file to push them into the database
    for file in files:
        file_path = os.path.join(docs_folder, file)
        print(f"📄 Processing {file}...")
        # This method handles text splitting and embedding generation
        db_manager.add_document(file_path)
    
    print("✅ The RAG Database (ChromaDB) is now ready to use!")

# Standard entry point to run the script
if __name__ == "__main__":
    run_ingestion()