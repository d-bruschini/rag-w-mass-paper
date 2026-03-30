import os
from openai import OpenAI
import pickle
import numpy as np

# Ensure OpenAI API key is set in environment variable
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

client = OpenAI(api_key=None)  # picks key from environmental variable

def create_embeddings(texts):
    """
    Generate embeddings for a list of texts using the OpenAI API.

    Args:
        texts (list of str): The texts to generate embeddings for.

    Returns:
        np.ndarray: A NumPy array containing the embeddings for each text.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small", 
            input=texts
        )
        embeddings = [e.embedding for e in response.data]
        return np.array(embeddings)
    except Exception as e:
        raise RuntimeError(f"Error while generating embeddings: {e}")

def save_embeddings(embeddings, path):
    """
    Save embeddings to a file.

    Args:
        embeddings (np.ndarray): The embeddings to save.
        path (str): The path to the file where embeddings should be saved.
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings successfully saved to {path}")
    except Exception as e:
        raise RuntimeError(f"Error while saving embeddings to {path}: {e}")

def load_embeddings(path):
    """
    Load embeddings from a file.

    Args:
        path (str): The path to the file containing embeddings.

    Returns:
        np.ndarray: The embeddings loaded from the file.

    Raises:
        FileNotFoundError: If the embeddings file does not exist.
        ValueError: If the embeddings file is corrupted or in an unexpected format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found at {path}. Please generate embeddings first.")
    
    try:
        with open(path, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Embeddings successfully loaded from {path}")
        return embeddings
    except Exception as e:
        raise ValueError(f"Error loading embeddings from {path}: {e}")
