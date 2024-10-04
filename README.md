# Voice conversational assistant with RAG

## Overview

This project implements a voice-enabled conversational assistant designed to answer spoken questions based on a set of provided documents. It functions as a voice-driven Retrieval-Augmented Generation (RAG) system, capable of processing voice queries and providing answers by retrieving relevant information from a collection of PDF documents. The assistant leverages the LangChain framework and integrates open-source models for advanced natural language processing tasks.

## Features

- STT: Speech recognition using the Whisper model.

- Retriever: Document retrieval from a vector store.

- Reranker: Rerank retrieved documents.

- Grader: Grade retrieved documents for relevance.

- Web Searching: Search the web for documents when no relevant information is found in the existing database.

- Generator: Generate an answer using a large language model (LLM).

- Rephraser: Rephrase the query for improved accuracy (coming soon).

- TTS: Convert generated answers to audio using Coqui (coming soon).

- Multi-modality: Extract unstructured data from PDFs and retrieve information in various formats (e.g., images, tables) (coming soon).

- LLM Judge: Verify the accuracy of generated responses (coming soon).

## Requirements

- Python 3.9+

## Installation

1. Clone the repository:
```
git clone https://github.com/scho/voice-enabled-rag.git
cd voice-enabled-rag
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Set up your environment variables:

Create a .env file in the root of the project and add your Hugging Face API key:
```
HF_API_KEY={your_huggingface_api_key}
```

## Usage
You can use either the graphical interface (demo.py) or run it without the interface (main.py):
### Option 1: Running with Interface
To use the graphical interface with Gradio:
```
python src/demo.py
```
### Option 2: Running without Interface
1. Run the main application:
```
python src/main.py
```
2. Speak your query when prompted. The application will process your input and provide a generated response based on the defined workflow.