import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --- Phase 1: Ingestion & Processing ---
def load_and_process_url(url: str):
    """
    Loads content from a URL using UnstructuredURLLoader, cleans it, and splits it into chunks.
    """
    try:
        # Initialize the loader with the specific URL
        loader = UnstructuredURLLoader(urls=[url])
        
        # Load the data
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content found at the provided URL.")
            
        # Split the text into chunks
        # chunk_size=1000: Max characters per chunk
        # chunk_overlap=200: Overlap to maintain context between chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    except Exception as e:
        st.error(f"Error loading URL: {str(e)}")
        return None

# --- Phase 2: Vector Storage ---
def setup_vectorstore(chunks):
    """
    Generates embeddings for the chunks using local HuggingFace embeddings and stores them in a local Chroma vector store.
    """
    import chromadb
    
    # Directory to persist the database
    persist_directory = "./chroma_db"
    
    # Initialize the PersistentClient
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Clear existing collection to avoid mixing data from different URLs
    # We use delete_collection instead of rmtree to avoid corrupting the client state
    try:
        client.delete_collection("langchain")
    except ValueError:
        # Collection might not exist, which is fine
        pass
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create the vector store using the existing client
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="langchain"
    )
    
    return vectorstore

# --- Phase 3: Conversational Interface (Logic Upgrade) ---
def get_rag_chain(vectorstore, api_key):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # 1. Logic-Focused LLM (Low Temp)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite-preview-09-2025", # Keep my working model
        google_api_key=api_key,
        temperature=0.0
    )

    # 2. "Dermatologist" Logic Prompt
    template = """You are a helpful skin expert. Use the context below to answer the question.
    
    RULES:
    1. If one product is for "AM" and another for "PM", explicitly state they should be separated.
    2. Check for conflicts (e.g., Don't mix Vitamin C with AHAs/BHA/Mandelic).
    3. If the answer is not in the context, say "I don't know".
    
    Context:
    {context}
    
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # 3. Enhanced Retrieval (k=7)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 7}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# --- Phase 4: Main Application ---
def main():
    st.set_page_config(page_title="Skin Expert", page_icon="🌐")
    st.title("🌐 Skin Expert")
    st.caption("Enter a URL and ask questions about its content!")

    # Sidebar for URL input
    with st.sidebar:
        st.header("Configuration")
        url_input = st.text_input("Enter Website URL")
        process_button = st.button("Process URL")
        
        if "GOOGLE_API_KEY" not in os.environ:
             # Try getting it from the environment directly in case it wasn't picked up
             env_key = os.getenv("GOOGLE_API_KEY")
             if env_key:
                 os.environ["GOOGLE_API_KEY"] = env_key
             else:
                 st.error("❌ GOOGLE_API_KEY not found in .env file. Please check your configuration.")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize session state for the QA chain
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # Process URL when button is clicked
    if process_button and url_input:
        if "GOOGLE_API_KEY" not in os.environ:
            st.error("Google API Key is required!")
        else:
            api_key = os.environ["GOOGLE_API_KEY"]
            with st.spinner("Processing website content..."):
                # Phase 1: Ingest
                chunks = load_and_process_url(url_input)
                
                if chunks:
                    st.success(f"Successfully loaded {len(chunks)} chunks from the website.")
                    
                    try:
                        # Phase 2: Vector Store
                        vectorstore = setup_vectorstore(chunks)
                        
                        # Phase 3: Setup QA Chain (Using upgraded logic)
                        st.session_state.qa_chain = get_rag_chain(vectorstore, api_key)
                        
                        # Clear chat history when new URL is processed
                        st.session_state.messages = []
                        st.success("Ready to chat!")
                    except Exception as e:
                        st.error(f"Error initializing Google Gemini components: {str(e)}")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the website"):
        if not st.session_state.qa_chain:
            st.warning("Please process a URL first.")
        else:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Check for greetings
            greetings = ["hello", "hi", "hey", "greetings"]
            contact_keywords = ["contact", "email", "phone", "number", "whatsapp"]
            
            if prompt.lower().strip() in greetings:
                greeting_response = "Hello! 👋 I am your Jenpharm AI assistant. How can I help you with your skincare routine today?"
                with st.chat_message("assistant"):
                    st.markdown(greeting_response)
                st.session_state.messages.append({"role": "assistant", "content": greeting_response})
            
            elif any(keyword in prompt.lower() for keyword in contact_keywords):
                contact_response = (
                    "**Email:** cs@jenpharm.com\n\n"
                    "**Phone/WhatsApp:** 03-111-444-536\n\n"
                    "**Address:** 254 H1 Johar Town, Lahore."
                )
                with st.chat_message("assistant"):
                    st.markdown(contact_response)
                st.session_state.messages.append({"role": "assistant", "content": contact_response})

            else:
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.qa_chain.invoke({"query": prompt})
                            answer = response["result"]
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
