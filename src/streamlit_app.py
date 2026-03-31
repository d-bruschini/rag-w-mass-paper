import os
from main import load_data, initialize_llm, maintain_context
from openai import OpenAI
from retrieval import create_context
import streamlit as st

# Initialize OpenAI client
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

client = OpenAI(api_key=None) # picks key from environmental variable

# Initialize LLM and chunks (and prevent from reloading everytime)
@st.cache_resource
def start_rag():
    split_docs, index = load_data()
    llm = initialize_llm(client)
    return llm, split_docs, index

rag, split_docs, index = start_rag()

# Title
st.title("RAG System for Scientific Document Analysis (CMS W Boson Mass)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Question: "):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call your RAG pipeline
    retrieved_information = create_context(prompt, index, split_docs)
    response = rag.response(prompt, retrieved_information, streaming=True)

    # iterate through the stream of events and display llm response
    placeholder = st.empty()
    collected_messages = []
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        filtered_messages = [m for m in collected_messages if m is not None]
        response_content = ''.join(filtered_messages)
        placeholder.markdown(response_content)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_content})

    # handle history to pass to LLM (at the moment it is just stored, not passed)
    history = list(st.session_state.messages)
    history = maintain_context(history)
