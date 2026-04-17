import os
import logging
from openai import OpenAI
from llm import LLM
from file_loader import load_and_split_pdf, save_split_docs, load_split_docs
from embeddings import create_embeddings, save_embeddings, load_embeddings
from retrieval import build_index, create_context
from config import EMBEDDINGS_PATH, MAX_HISTORY_LENGTH, FILE_PATH

logging.basicConfig(level=logging.INFO)

def maintain_history(history):
    if len(history) > MAX_HISTORY_LENGTH * 2:  # 2 for question and answer pairs
        history = history[-(MAX_HISTORY_LENGTH * 2):]
    return history

def load_data():
    '''
    Embeddings and split documents are already available for the W mass paper with the default settings. However, if you change some of the parameters in config.py,
    particularly those regarding the chunk size and overlap, or you want to use this chatbot with another paper, you need to regenerate the embeddings, and split the document again
    (unless you decide to provide the name of a PDF file in FILE_PATH, in which case the splitting will be performed automatically).
    For this, two functions are available in embeddings.py, create_embeddings(texts), which takes as input the chunks obtained by splitting the file,
    and save_embeddings(embeddings, path), where embeddings is the output of create_embeddings(texts) and path is the name of the pickle file where the new embeddings are stored.
    An analogous function is available in file_loader.py, save_split_docs(split_docs, path), where split docs is the output of load_and_split_pdf()
    Make sure to update the EMBEDDINGS_PATH and FILE_PATH in config.py accordingly.
    '''

    if FILE_PATH.endswith('.pdf'):
        split_docs = load_and_split_pdf()
    else:
        split_docs = load_split_docs(FILE_PATH)
    chunk_texts = [chunk.page_content for chunk in split_docs] #needed to recreate embeddings

    embeddings = load_embeddings(EMBEDDINGS_PATH)
    index = build_index(embeddings)
    return split_docs, index

def initialize_llm(client):
    logging.info("Initializing LLM...")
    return LLM(client)

def start_chatbot(llm, split_docs, index):
    print("Chatbot is now running. Type 'exit' or 'quit' to end the session.")
    history = [] # Chat history is currently stored, but not used as information for the response generation. Should you want to do this, pass the 'history=history' argument to llm.response
    while True:
        question = input("Question: ")
        if question.lower() in ["exit", "quit"]:
            break
        retrieved_information = create_context(question, index, split_docs)
        history = maintain_history(history)
        response = llm.response(question, retrieved_information)
        response_content = response.choices[0].message.content.strip() if response.choices else "Sorry, I couldn't generate an answer."
        print(f"Chatbot: {response_content}")
        history.append({"role":"user", "content": question})
        history.append({"role":"assistant", "content": response_content})

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

    client = OpenAI(api_key=None) # picks key from environmental variable
    split_docs, index = load_data()
    llm = initialize_llm(client)
    start_chatbot(llm, split_docs, index)
