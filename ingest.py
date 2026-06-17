import os
import logging
from dotenv import load_dotenv

from src.data_loader import load_documents
from src.chunker import split_documents
from src.embedder import get_embedding_model
from src.vector_store import create_or_update_vector_store

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")

def main():
    """
    Main function to run the data ingestion pipeline.
    
    This script coordinates the loading of Meezan Bank documents, splitting them
    into manageable chunks, generating sentence-transformer embeddings, and
    saving the embedded chunks into a persistent ChromaDB vector store.
    """
    logger.info("Starting data ingestion process...")
    
    # 1. Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        logger.info(f"Creating data directory at {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.warning("Data directory was empty/missing. Please place your PDF and TXT documents in 'data/documents' and run ingest.py again.")
        return

    # 2. Load Documents
    logger.info("Step 1: Loading documents...")
    documents = load_documents(DATA_DIR)
    if not documents:
        logger.warning("No valid documents found in data directory. Ingestion stopped.")
        return
        
    logger.info(f"Loaded {len(documents)} document page(s) successfully.")

    # 3. Split Documents into Chunks
    logger.info("Step 2: Splitting documents into chunks...")
    chunks = split_documents(documents)
    if not chunks:
        logger.warning("No chunks could be created from the documents. Ingestion stopped.")
        return

    # 4. Initialize Embedding Model
    logger.info("Step 3: Initializing embedding model...")
    try:
        embedding_model = get_embedding_model()
    except Exception as e:
        logger.error(f"Failed to initialize embedding model. Ingestion stopped. Error: {e}")
        return

    # 5. Create or Update Persistent Vector Store
    logger.info("Step 4: Storing chunks in ChromaDB...")
    try:
        # This function natively persists the database into DB_DIR
        create_or_update_vector_store(
            documents=chunks,
            embedding_model=embedding_model,
            persist_directory=DB_DIR
        )
        logger.info(f"Ingestion complete! Index successfully saved and persisted to {DB_DIR}.")
    except Exception as e:
        logger.error(f"Failed to create vector store. Ingestion stopped. Error: {e}")

if __name__ == "__main__":
    main()
