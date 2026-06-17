import os
import logging
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# Prompt template tailored for Meezan Bank context
RAG_PROMPT_TEMPLATE = """
You are a highly knowledgeable and professional AI Assistant for Meezan Bank.
Your task is to answer user queries based ONLY on the provided context, which includes
Meezan Bank's product guides, annual reports, Shariah policies, Islamic banking FAQs,
and financial statements.

If the answer is not contained within the provided context, politely say that you
cannot answer based on the available Meezan Bank documents. Do not invent or hallucinate answers.

Context:
{context}

User Query:
{question}

Answer:
"""

def initialize_llm() -> ChatGroq:
    """
    Initializes the Groq LLM using the provided API key from the environment.
    
    Returns:
        ChatGroq: An initialized instance of the LangChain ChatGroq model (Llama 3).
        
    Raises:
        ValueError: If GROQ_API_KEY is not found in environment variables.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY environment variable is not set or invalid.")
        
    try:
        # Using Llama 3.1 8B via Groq
        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {str(e)}")
        raise

def perform_search_and_answer(query: str, vector_store: Chroma, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieves the most relevant document chunks for a query and uses the Groq LLM
    to generate a context-aware answer.
    
    This function performs a similarity search with relevance scores, formats the
    retrieved context, constructs a prompt specifically for Meezan Bank, and
    invokes the LLM. It gracefully handles API failures and cases where no
    relevant documents are found.
    
    Parameters:
        query (str): The user's question.
        vector_store (Chroma): The persistent Chroma vector store to search.
        top_k (int): The number of most relevant chunks to retrieve. Default is 3.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'answer' (str): The LLM generated response.
            - 'source_file' (str): The filename of the most relevant source.
            - 'page_number' (int): The page number of the most relevant source.
            - 'confidence_score' (float): The similarity score of the most relevant chunk.
                                          Returns a default/empty dict if no results or error.
    """
    try:
        # Perform similarity search returning scores
        # Using similarity_search_with_score; Chroma returns distance (lower is closer)
        results = vector_store.similarity_search_with_score(query, k=top_k)
        
        if not results:
            logger.warning(f"No results found for query: {query}")
            return {
                "answer": "I could not find any relevant information in the provided Meezan Bank documents.",
                "source_file": "N/A",
                "page_number": "N/A",
                "confidence_score": 0.0
            }

        # Extract context and metadata from the top chunks
        context_texts = []
        for doc, _ in results:
            context_texts.append(doc.page_content)
            
        context_str = "\n\n---\n\n".join(context_texts)
        
        # Initialize LLM and Prompt
        llm = initialize_llm()
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Chain invocation using new LangChain Expression Language (LCEL)
        chain = prompt | llm
        
        logger.info(f"Invoking Groq LLM for query: {query}")
        response = chain.invoke({"context": context_str, "question": query})
        answer = response.content
        
        # Determine the most relevant source (the first one)
        top_doc, top_score = results[0]
        
        return {
            "answer": answer,
            "source_file": top_doc.metadata.get("source_filename", "Unknown"),
            "page_number": top_doc.metadata.get("page", "Unknown"),
            # For Chroma, score is distance, we can just return it or invert it.
            # Returning raw score, note that lower is usually better for distance.
            "confidence_score": float(top_score)
        }

    except Exception as e:
        logger.error(f"Error during search and LLM invocation: {str(e)}")
        return {
            "answer": f"I encountered an error while trying to process your request: {str(e)}",
            "source_file": "Error",
            "page_number": "Error",
            "confidence_score": 0.0
        }
