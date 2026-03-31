import os
import logging
from openai import OpenAI
from llm import LLM
from file_loader import load_and_split_pdf
from embeddings import create_embeddings, save_embeddings, load_embeddings
from retrieval import build_index, create_context
from config import EMBEDDINGS_PATH, MAX_CONTEXT_LENGTH

logging.basicConfig(level=logging.INFO)

def maintain_context(context):
    if len(context) > MAX_CONTEXT_LENGTH * 2:  # 2 for question and answer pairs
        context = context[-(MAX_CONTEXT_LENGTH * 2):]
    return context

def load_data():
    logging.info("Loading and processing paper...")
    split_docs = load_and_split_pdf()
    chunks = [chunk.page_content for chunk in split_docs] #needed to recreate embeddings
    '''
    Embeddings are already available for the W mass paper with the default settings. However, if you change some of the parameters in config.py,
    particularly those regarding the chunk size and overlap, or you want to use this chatbot with another paper, you need to regenerate the embeddings.
    For this, two functions are available in embeddings.py, create_embeddings(texts), which takes as input the chunks obtained by splitting the file,
    and save_embeddings(embeddings, path), where embeddings is the output of create_embeddings(texts) and path is the name of the pickle file where the new embeddings are stored.
    Make sure to update the EMBEDDINGS_PATH in config.py accordingly.
    '''

    embeddings = load_embeddings(EMBEDDINGS_PATH)
    index = build_index(embeddings)
    return split_docs, index

def initialize_llm(client):
    logging.info("Initializing LLM...")
    return LLM(client)

def start_chatbot(llm, split_docs, index):
    print("Chatbot is now running. Type 'exit' or 'quit' to end the session.")
    context = []
    while True:
        question = input("Question: ")
        if question.lower() in ["exit", "quit"]:
            break
        retrieved_information = create_context(question, index, split_docs)
        response = llm.response(question, retrieved_information)
        context = maintain_context(context)
        response_content = response.choices[0].message.content.strip() if response.choices else "Sorry, I couldn't generate an answer."
        print(f"Chatbot: {response_content}")
        context.append({"role":"user", "content": question})
        context.append({"role":"assistant", "content": response_content})

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

    client = OpenAI(api_key=None) # picks key from environmental variable
    split_docs, index = load_data()
    llm = initialize_llm(client)
    start_chatbot(llm, split_docs, index)
