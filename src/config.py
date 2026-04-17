import os

# API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

# Data
FILE_PATH = os.getenv("FILE_PATH", "../data/W_mass_chunks.pkl") # Path to the scientific splitted document (by default W mass splitted PDF). This can either be a PDF file, in which case the document will be split, or a pkl files already with the split documents
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "../data/W_mass_embeddings.pkl") # Path to embeddings file
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
TOP_K = 20

# Chatbot (OpenAI API)
MODEL = os.getenv("MODEL", "gpt-4.1-mini")  # Default to the current model if not set
TEMPERATURE = 0 # Controls the randomness of the LLM output (0 for deterministic, higher for more creative responses)
MAX_HISTORY_LENGTH = 5  # Number of Q&A pairs to keep in history
