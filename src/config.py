import os

# API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

# Data
PDF_PATH = os.getenv("PDF_PATH", "../data/W_mass_paper.pdf") # Path to the W mass paper (PDF)
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "../data/W_mass_embeddings.pkl") # Path to embeddings file
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
TOP_K = 20

# Chatbot (OpenAI API)
MODEL = os.getenv("MODEL", "gpt-5.4-2026-03-05")  # Default to the current model if not set
TEMPERATURE = 0 # Controls the randomness of the LLM output (0 for deterministic, higher for more creative responses)
MAX_CONTEXT_LENGTH = 5  # Number of Q&A pairs to keep in context
