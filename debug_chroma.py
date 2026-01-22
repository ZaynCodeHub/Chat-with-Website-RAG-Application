import os
import shutil
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

def test_chroma_setup():
    persist_directory = "./chroma_db"
    
    print(f"Testing ChromaDB version: {chromadb.__version__}")
    
    # Simulate the app logic: delete if exists
    if os.path.exists(persist_directory):
        print(f"Removing existing directory: {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # Dummy chunks
    chunks = ["This is a test chunk.", "Another chunk for testing."]
    
    # Embeddings
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vectorstore
    print("Creating vectorstore (from_documents)...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks, # In reality these are Document objects but strings might work or error differently
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print("Vectorstore created successfully.")
        
        # Verify persistence
        print(f"Directory exists: {os.path.exists(persist_directory)}")
        
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chroma_setup()
