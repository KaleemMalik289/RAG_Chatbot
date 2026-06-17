import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Initializes and returns a Sentence-Transformer embedding model.
    
    This function sets up the HuggingFaceEmbeddings class using the specified
    sentence-transformers model. It is solely responsible for providing the
    embedding logic required to convert text chunks into vector representations.
    
    Parameters:
        model_name (str): The name of the sentence-transformers model to use.
                          Defaults to 'all-MiniLM-L6-v2' as requested.
                          
    Returns:
        HuggingFaceEmbeddings: An initialized LangChain embedding model instance ready
                               to generate text embeddings.
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        
        # Initialize the HuggingFace embeddings via LangChain
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        logger.info("Embedding model loaded successfully.")
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {str(e)}")
        raise
