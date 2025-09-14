import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(data_folder):
    """Load all documents from the data folder"""
    documents = []
    
    # Get all files in data folder
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        
        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ Loaded PDF: {filename}")
                
            elif filename.endswith('.txt'):
                loader = TextLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ Loaded text file: {filename}")
                
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    return documents

def split_documents(documents):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Size of each chunk
        chunk_overlap=200,  # Overlap between chunks
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks