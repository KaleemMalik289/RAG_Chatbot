import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

def create_or_update_vector_store(
    documents: List[Document], 
    embedding_model: Embeddings, 
    persist_directory: str = "./vector_db"
) -> Chroma:
    """
    Creates a new persistent ChromaDB vector store or updates an existing one
    with the provided document chunks and their embeddings.
    
    This function handles the initialization of ChromaDB, configures it to be
    persistent (so data survives restarts), and ingests the document chunks.
    
    Parameters:
        documents (List[Document]): The chunked documents to be stored in the vector database.
        embedding_model (Embeddings): The LangChain embedding model used to generate vectors.
        persist_directory (str): The directory where ChromaDB will persist its data on disk.
                                 Defaults to './vector_db'.
                                 
    Returns:
        Chroma: An instance of the persistent LangChain Chroma vector store.
    """
    try:
        logger.info(f"Initializing persistent ChromaDB at {persist_directory}")
        
        # Chroma.from_documents automatically handles creation and persistent storage 
        # when a persist_directory is provided. (Auto-persists in ChromaDB >= 0.4.0)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        
        logger.info(f"Successfully saved {len(documents)} chunks to the vector store at {persist_directory}.")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating/updating vector store: {str(e)}")
        raise

def load_vector_store(embedding_model: Embeddings, persist_directory: str = "./vector_db") -> Chroma:
    """
    Loads an existing persistent ChromaDB vector store from disk.
    
    This function connects to a previously created Chroma vector store allowing
    for searches and retrievals without re-ingesting the documents.
    
    Parameters:
        embedding_model (Embeddings): The LangChain embedding model to use for querying.
                                      Must be the same model used during creation.
        persist_directory (str): The directory where ChromaDB persists its data.
                                 Defaults to './vector_db'.
                                 
    Returns:
        Chroma: An instance of the persistent LangChain Chroma vector store.
    """
    if not os.path.exists(persist_directory):
        logger.warning(f"Vector store directory '{persist_directory}' does not exist. It may need to be created first.")
        
    try:
        logger.info(f"Loading persistent ChromaDB from {persist_directory}")
        
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        
        logger.info("Vector store loaded successfully.")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise
