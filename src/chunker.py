import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks using
    RecursiveCharacterTextSplitter.
    
    This function processes each document, splits its text into manageable chunks,
    and preserves existing metadata (like source filename and page number) while
    adding a specific 'chunk_index' to track the chunk's order within the original text.
    
    Parameters:
        documents (List[Document]): The list of loaded documents to be split.
        chunk_size (int): The maximum number of characters for each chunk. Default is 1000.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks. Default is 200.
        
    Returns:
        List[Document]: A new list of LangChain Document objects representing the chunks.
                        Returns an empty list if the input documents list is empty.
    """
    if not documents:
        logger.warning("No documents provided to split.")
        return []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_documents = []
        
        for doc in documents:
            # Skip empty documents
            if not doc.page_content.strip():
                continue
                
            # Split the single document into chunks
            chunks = text_splitter.split_documents([doc])
            
            # Add chunk_index to metadata for tracking
            for i, chunk in enumerate(chunks):
                # Ensure we have a separate metadata dict for the chunk
                chunk.metadata = chunk.metadata.copy()
                chunk.metadata['chunk_index'] = i
                chunked_documents.append(chunk)
                
        logger.info(f"Successfully split {len(documents)} documents into {len(chunked_documents)} chunks.")
        return chunked_documents
        
    except Exception as e:
        logger.error(f"Error occurred while splitting documents: {str(e)}")
        return []
