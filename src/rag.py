import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RAG:
    def __init__(self, input_path) -> None:
        self.loader = PyPDFDirectoryLoader(input_path)
        self.data = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
            encoding_name='cl100k_base'
        )
        self.docs = self.text_splitter.split_documents(self.data)
        self.embeddings = SentenceTransformerEmbeddings()
        self.vector_store = Chroma.from_documents(
            self.docs, 
            self.embeddings,
            collection_name = 'renault',
            persist_directory = './db/chromadb',
        )

    def retriever(self, query):
        res = self.vector_store.similarity_search(
            query,
            k=2,
            filter={"source": "tweet"},
        )
        return res
