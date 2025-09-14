import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

def create_vector_store(chunks):
    """Create vector store from document chunks"""
    
    # Initialize embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"✅ Created vector store with {len(chunks)} documents")
    return vectorstore

def load_vector_store():
    """Load existing vector store"""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    return vectorstore