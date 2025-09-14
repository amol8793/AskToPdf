from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import openai

def create_chat_chain(vectorstore, openai_api_key):
    """Create the conversational chain"""
    
    # Initialize OpenAI
    llm = OpenAI(
        temperature=0.7,
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo-instruct"
    )
    
    # Create memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return chain

def get_response(chain, question):
    """Get response from the chain"""
    try:
        result = chain({"question": question})
        return result["answer"], result.get("source_documents", [])
    except Exception as e:
        return f"Error: {e}", []