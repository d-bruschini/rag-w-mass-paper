import os
import logging
from openai import OpenAI
from chatbot import Chatbot
from file_loader import load_and_split_pdf
from embeddings import create_embeddings, save_embeddings, load_embeddings
from retrieval import build_index
from config import EMBEDDINGS_PATH

logging.basicConfig(level=logging.INFO)

def format_pages(pages):
    roman_map = {
        -1: "i",
        0: "ii"
    }
    
    return [
        roman_map.get(page - 1, page - 1)
        for page in pages
    ]

def load_data():
    logging.info("Loading and processing paper...")
    chunks = load_and_split_pdf()
    title = chunks[0].metadata.get("title","unknown")
    pages = [chunk.metadata.get("page","unknown") for chunk in chunks]
    pages = format_pages(pages) # necessary formatting to make sure that the page numbers returned by the LLM are the same as in the paper
    chunks = [chunk.page_content for chunk in chunks]
    '''
    Embeddings are already available for the W mass paper with the default settings. However, if you change some of the parameters in config.py,
    particularly those regarding the chunk size and overlap, or you want to use this chatbot with another paper, you need to regenerate the embeddings.
    For this, two functions are available in embeddings.py, create_embeddings(texts), which takes as input the chunks obtained by splitting the file,
    and save_embeddings(embeddings, path), where embeddings is the output of create_embeddings(texts) and path is the name of the pickle file where the new embeddings are stored.
    Make sure to update the EMBEDDINGS_PATH in config.py accordingly.
    '''

    embeddings = load_embeddings(EMBEDDINGS_PATH)
    index = build_index(embeddings)
    return chunks, pages, index, title

def initialize_chatbot(client, index, chunks, pages, title):
    logging.info("Initializing chatbot...")
    return Chatbot(client, index, chunks, pages, title)

def start_chatbot(chatbot):
    print("Chatbot is now running. Type 'exit' or 'quit' to end the session.")
    while True:
        question = input("Question: ")
        if question.lower() in ["exit", "quit"]:
            break
        chatbot.chat(question)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

    client = OpenAI(api_key=None) # picks key from environmental variable
    chunks, pages, index, title = load_data()
    chatbot = initialize_chatbot(client, index, chunks, pages, title)
    start_chatbot(chatbot)
