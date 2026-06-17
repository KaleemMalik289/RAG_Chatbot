import os
import sys
from pathlib import Path

# Force the absolute path of the project root into sys.path
PROJECT_ROOT = str(Path(__file__).parent.absolute())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Workaround for Python caching a global 'src' module (common issue in Streamlit environments)
if "src" in sys.modules:
    del sys.modules["src"]
for key in list(sys.modules.keys()):
    if key.startswith("src."):
        del sys.modules[key]

import streamlit as st
from dotenv import load_dotenv

from src.embedder import get_embedding_model
from src.vector_store import load_vector_store
from src.search import perform_search_and_answer

# Load environment variables (GROQ_API_KEY)
load_dotenv()

# Constants
DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Meezan Bank AI Assistant",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def initialize_system():
    """
    Initializes the embedding model and vector store, caching them so they 
    don't reload on every user interaction.
    
    Returns:
        tuple: (embedding_model, vector_store) or (None, None) if an error occurs.
    """
    try:
        embedding_model = get_embedding_model()
        
        # Check if vector db exists (checking for ChromaDB directory)
        if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
            st.error("Vector database not found. Please put some documents in data/documents and run `python ingest.py` first.")
            return None, None
            
        vector_store = load_vector_store(
            embedding_model=embedding_model, 
            persist_directory=DB_DIR
        )
        return embedding_model, vector_store
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None, None

def main():
    """
    Main function to run the Streamlit UI.
    
    It handles chat history, user input, invokes the search and retrieval
    functions, and beautifully displays the response along with metadata 
    (source, page, confidence).
    """
    st.title("🏦 Meezan Bank AI Assistant")
    st.markdown("Ask me questions about Meezan Bank's product guides, annual reports, Shariah policies, and FAQs!")

    # Initialize backend components
    embedding_model, vector_store = initialize_system()
    
    if not vector_store:
        st.stop()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message and message["metadata"]["source_file"] not in ["N/A", "Error"]:
                with st.expander("View Source Details"):
                    st.write(f"**Source File:** {message['metadata']['source_file']}")
                    st.write(f"**Page Number:** {message['metadata']['page_number']}")
                    st.write(f"**Confidence Score:** {message['metadata']['confidence_score']:.4f}")

    # Accept user input
    if prompt := st.chat_input("How can I help you today?"):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Searching Meezan Bank documents..."):
                result = perform_search_and_answer(
                    query=prompt, 
                    vector_store=vector_store, 
                    top_k=3
                )
                
                answer = result["answer"]
                source_file = result["source_file"]
                page_number = result["page_number"]
                confidence_score = result["confidence_score"]
                
                st.markdown(answer)
                
                # Display metadata visually if valid
                if source_file not in ["N/A", "Error"]:
                    with st.expander("View Source Details"):
                        st.write(f"**Source File:** {source_file}")
                        st.write(f"**Page Number:** {page_number}")
                        st.write(f"**Confidence Score:** {confidence_score:.4f}")
                        
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "metadata": {
                        "source_file": source_file,
                        "page_number": page_number,
                        "confidence_score": confidence_score
                    }
                })

if __name__ == "__main__":
    main()
