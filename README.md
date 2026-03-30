# RAG System for Scientific Document Analysis (CMS W Boson Mass)

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to interpret the [**latest W mass paper by the CMS experiment**](https://arxiv.org/abs/2412.13872). It enables users to query a complex high-energy physics paper in natural language and receive context-grounded, receive context-grounded, human-readable explanations with page-level source references.

The system implements a full RAG pipeline:

* **Document preprocessing**: The paper is segmented into manageable chunks.
* **Embedding**: Each chunk is converted into a dense vector representation.
* **Retrieval**: Relevant sections are identified via semantic similarity search.
* **Generation**: Answers are produced using an LLM, grounded in retrieved context.

This approach ensures that responses are relevant, interpretable, and anchored to the source document.

## Architecture

* **Chunking**: Document segmentation using LangChain.
* **Embeddings**: Dense vector representations generated via the OpenAI API.
* **Vector Search**: Semantic retrieval implemented with FAISS.
* **LLM Generation**: Context-aware answer generation using OpenAI models.

The system constrains the language model to use only retrieved context, helping reduce hallucinations and improve factual consistency.

## Features

* **Semantic Search over Scientific Text**: Retrieves the most relevant sections of the paper based on query meaning.
* **Context-Grounded Responses**: Answers are generated strictly from retrieved content.
* **Configurable Pipeline**: Adjustable parameters such as:
	* chunk_size and overlap (document segmentation)
	* top_k (number of retrieved chunks)
	* temperature (generation randomness)
* **Lightweight Interface**: Simple command-line interaction for rapid experimentation.
* **Citation-Aware RAG Pipeline**: The system provides context-grounded answers with explicit references (page numbers) to the original scientific document, improving transparency and verifiability.

## Installation

**Prerequisites**

* Python 3.7+
* An OpenAI API key (for access to GPT models)

Step 1: Clone the repository

```
git clone https://github.com/d-bruschini/rag-w-mass-paper.git
cd rag-w-mass-paper
```

Step 2: Set up virtual environment (to run only the first time)

```
python3 -m venv venv
```

Step 3: Activate virtual environment

```
source venv/bin/activate
```

Step 4: Install dependencies

```
pip install -r requirements.txt
```

Step 5: Set up OpenAI API Key

Ensure that your OpenAI API key is set as an environment variable:

```
export OPENAI_API_KEY="your-openai-api-key"
```

Step 6: Move to src/

```
cd src/
```

## Usage

The W mass paper is already available in data/ in PDF format and the chatbot can readily be used by running

```
python3 main.py
```

You can now interact with the chatbot directly from the command line.

### Regenerating Embeddings

If you change the parameters in config.py, especially chunk_size and overlap, or you wish to use a different paper, you need to regenerate the embeddings. The embeddings store the vector representations of the text chunks, and if the chunks change the embeddings need to be regenerated to ensure accurate retrieval. To do so, use the functions in embeddings.py:

* create_embeddings(texts): Takes the chunked text and generates embeddings.
* save_embeddings(embeddings, path): Saves the embeddings to a pickle file for later use.

In config.py, you can modify parameters such as chunk_size, overlap, and the path to the embeddings file. Be sure to adjust these settings according to your needs.

### Other parameters

This chatbot is implemented using the OpenAI API. In the config.py file, you can also adjust the model used for output generation and fine-tune the temperature parameter, which controls the randomness of the generated text (higher values produce more creative responses, while lower values make the output more deterministic).

## Current limitations and possible improvements
* The parameters have been set to optimize the accuracy of the answers, but some responses may still be incorrect. In addition, the chunk size and top_k might currently be large.
* Tables and figures are not retrieved properly.
* Consider whether to include chat history (currently, it's stored but not passed to the LLM).
* Consider whether to use the LLM's internal knowledge (currently, the prompt explicitly indicates that only the context provided must be used).
* Future improvements could include better handling of figures/tables and refining chat history usage for more interactive conversations.
* No formal evaluation metric for answer quality (e.g., precision/recall of retrieval) yet.

## Key Takeaway

This project reframes a complex scientific document as a data problem, showing how modern ML techniques can enable efficient information retrieval from sources such as publications by CERN.
