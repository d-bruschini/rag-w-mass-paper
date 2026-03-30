from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_and_split_pdf(pdf_path=PDF_PATH):
    """
    Loads a PDF from the specified path and splits it into chunks using the RecursiveCharacterTextSplitter.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List of chunks (list of str): The chunks of text obtained after splitting the PDF.
    """
    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}. Please check the file path.")

    logging.info(f"Loading PDF from {pdf_path}")
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    logging.info(f"Loaded {len(docs)} pages from the PDF.")

    # Split the document into chunks using the configured separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    logging.info(f"Split PDF into {len(chunks)} chunks.")
    
    return chunks
