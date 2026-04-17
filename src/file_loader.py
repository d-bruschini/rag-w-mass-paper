from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP
import pickle
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_and_split_pdf(pdf_path=FILE_PATH):
    """
    Loads a PDF from the specified path and splits it into chunks using the RecursiveCharacterTextSplitter.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        List of split_docs (list of langchain_core.documents.base.Document): The document splitting obtained after splitting the PDF.
    """

    # Check if input name is PDF file
    if not pdf_path.endswith('.pdf'):
        raise TypeError(f"{pdf_path} must be a PDF file for \'load_and_split_pdf\' to work properly. Please check the file path.")

    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}. Please check the file path.")

    logging.info(f"Loading PDF from {pdf_path}")
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    logging.info(f"Loaded {len(docs)} pages from the PDF.")

    # Split the document into chunks using the configured separators
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    logging.info(f"Split PDF into {len(chunks)} chunks.")
    
    return chunks

def save_split_docs(docs, path):
    """
    Save split documents to a file.

    Args:
        docs (list of langchain_core.documents.base.Document): The documents to save.
        path (str): The path to the file where documents should be saved.
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(docs, f)
        print(f"Split documents successfully saved to {path}")
    except Exception as e:
        raise RuntimeError(f"Error while saving split documents to {path}: {e}")

def load_split_docs(path):
    """
    Load split docs from a file.

    Args:
        path (str): The path to the file containing split docs.

    Returns:
        list of langchain_core.documents.base.Document: The split docs loaded from the file.

    Raises:
        FileNotFoundError: If the split docs file does not exist.
        ValueError: If the split docs file is corrupted or in an unexpected format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split docs file not found at {path}. Please generate split docs first.")
    
    try:
        with open(path, "rb") as f:
            split_docs = pickle.load(f)
        print(f"Split docs successfully loaded from {path}")
        return split_docs
    except Exception as e:
        raise ValueError(f"Error loading split docs from {path}: {e}")
