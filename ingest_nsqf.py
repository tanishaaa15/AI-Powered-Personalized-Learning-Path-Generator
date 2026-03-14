import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuration
DATA_PATH = "data"           # Folder where your NSQF PDFs are kept
DB_PATH = "chroma_db"        # Folder where the database will be saved

def run_ingestion():
    # 2. Check for PDFs
    if not os.path.exists(DATA_PATH) or not any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        print(f"❌ Error: No PDFs found in '{DATA_PATH}' folder. Put your NSQF files there first!")
        return

    # 3. Load PDFs
    print("🔄 Loading PDFs...")
    loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages.")

    # 4. Split text into chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")

    # 5. Create Embeddings & Store in ChromaDB
    print("🧠 Generating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # This creates the 'chroma_db' folder and saves the data
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print(f"✨ SUCCESS! Database created at '{DB_PATH}' with {len(chunks)} chunks.")

if __name__ == "__main__":
    run_ingestion()
