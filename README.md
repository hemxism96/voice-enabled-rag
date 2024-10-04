# Voice conversational assistant with RAG

This project implements a voice conversational assistant. Thea ssistant is intended to reply to voice questions based on given documents. It's a voice-enabled RAG system capable of answering spoken questions based on the information contained within a provided set of PDF documents. It utilizes the LangChain framework and integrates with open source models for advanced natural language processing tasks.

## Requirements

- Python 3.9.20

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
HF_API_KEY=your_huggingface_api_key
```

## Usage
1. Run the main application:
```
python main.py
```
2. Speak your query when prompted. The application will process your input and provide a generated response based on the defined workflow.

## Features

- Speech recognition using the Whisper model.

- Document retrieval from a vector store.

- Rerank retrieved documents.

- Grading of retrieved documents for relevance.

- Web searching in case there is no relevant information from given documents.

- Generation of answers using a large language model.

- Query rephrasing for improved accuracy. (coming)

- TTS: convert generated answer by coqui (coming)

- multi-modality: unstructure pdf and get information in many format features. ex) image, tables (coming)

- LLM judge: verify generated response (coming)