import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_documents_from_folder(folder_path):
    """Load all text files from data folder"""
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        st.error(f"Folder '{folder_path}' not found!")
        return []
    
    # Load all .txt files
    for file_path in folder.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append({
                    'filename': file_path.name,
                    'content': content
                })
        except Exception as e:
            st.error(f"Error loading {file_path.name}: {e}")
    
    return documents

def main():
    st.title("🤖 RAG Chatbot - Document Q&A")
    st.write("Ask questions about your documents!")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please add your GOOGLE_API_KEY to the .env file")
        st.info("Get your key at: https://aistudio.google.com/")
        return
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Load documents
        documents = load_documents_from_folder("data")
        
        if documents:
            st.success(f"✅ Found {len(documents)} documents:")
            for doc in documents:
                st.write(f"• {doc['filename']}")
        else:
            st.warning("No .txt files found in 'data' folder!")
            st.info("Add .txt files to the 'data' folder and refresh")
            return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking..."):
                try:
                    # Combine all document content
                    all_content = "\n\n".join([
                        f"Document: {doc['filename']}\n{doc['content']}" 
                        for doc in documents
                    ])
                    
                    # Create prompt with documents and question
                    full_prompt = f"""
Based on the following documents, please answer the user's question. If the answer is not in the documents, please say so.

DOCUMENTS:
{all_content}

USER QUESTION: {prompt}

Please provide a helpful and accurate answer based on the information in the documents above.
"""
                    
                    # Get response from Gemini
                    response = model.generate_content(full_prompt)
                    ai_response = response.text
                    
                    st.markdown(ai_response)
                    
                    # Show source documents
                    with st.expander("📄 Source Documents Used"):
                        for doc in documents:
                            st.write(f"**{doc['filename']}**")
                            preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                            st.write(preview)
                            st.write("---")
                    
                except Exception as e:
                    ai_response = f"Sorry, I encountered an error: {e}"
                    st.error(ai_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

if __name__ == "__main__":
    main()