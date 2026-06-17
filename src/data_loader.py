import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents(directory_path: str) -> List[Document]:
    """
    Loads all PDF and TXT documents from a specified directory.
    
    This function scans the given directory for .pdf and .txt files,
    loads them using appropriate LangChain loaders, and extracts their
    content along with metadata (source filename and page number).
    
    Parameters:
        directory_path (str): The absolute or relative path to the directory containing the documents.
        
    Returns:
        List[Document]: A list of LangChain Document objects loaded from the files.
                        Returns an empty list if no files are found or if an error occurs.
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return []

    documents = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Skip if it is a directory
        if not os.path.isfile(file_path):
            continue
            
        try:
            if filename.lower().endswith('.pdf'):
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                
                if not docs:
                    logger.warning(f"Empty or unreadable PDF: {filename}")
                else:
                    for doc in docs:
                        # PyMuPDFLoader adds 'page' (0-indexed). Let's make it 1-indexed for readability
                        page_num = doc.metadata.get('page', 0) + 1
                        doc.metadata['page'] = page_num
                        doc.metadata['source_filename'] = filename
                    
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} pages from {filename}")
                    
            elif filename.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                # Check for empty text documents
                docs = [doc for doc in docs if doc.page_content.strip()]
                
                if not docs:
                    logger.warning(f"Empty TXT document: {filename}")
                else:
                    for doc in docs:
                        doc.metadata['page'] = 1  # TXT files are treated as single page
                        doc.metadata['source_filename'] = filename
                        
                    documents.extend(docs)
                    logger.info(f"Loaded TXT file: {filename}")
                    
        except Exception as e:
            logger.error(f"Failed to load file {filename}: {str(e)}")
            continue

    if not documents:
        logger.warning(f"No valid documents could be extracted from {directory_path}")
        
    return documents
