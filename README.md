# RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **ChromaDB**. It allows users to query information based on uploaded PDF and text documents using state-of-the-art embedding models.

## 🚀 Live Demo

You can access the deployed application here: 
**[Chatbot Live Deployment](https://chatttbott.streamlit.app/)**

## 🛠️ Technologies Used

- **Streamlit**: Web interface for the chatbot.
- **LangChain**: Framework for developing LLM-driven applications.
- **ChromaDB**: Vector database for storing and querying document embeddings.
- **HuggingFace & SentenceTransformers**: For generating high-quality text embeddings (`all-MiniLM-L6-v2`).
- **PyMuPDF**: For robust PDF document parsing and text extraction.
- **Groq API**: High-performance LLM inference.

## 📂 Project Structure

- `app.py`: Main Streamlit application file containing the UI and chatbot logic.
- `ingest.py`: Script used to process documents, chunk them, generate embeddings, and store them in the vector database.
- `src/`: Contains core modules like chunking, document loading, and vector store management.
- `data/documents/`: Directory where PDF and TXT files are stored for ingestion.
- `vector_db/`: Local persistent storage for the ChromaDB vector database.

## 💻 Local Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your environment variables (such as your Groq API key) to a `.env` file.
4. Run the data ingestion to populate the database:
   ```bash
   python ingest.py
   ```
5. Start the Streamlit server:
   ```bash
   python -m streamlit run app.py
   ```
