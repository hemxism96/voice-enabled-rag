from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

loader = PyPDFDirectoryLoader("./data")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
docs = text_splitter.split_documents(data)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)
vector_store = Chroma.from_documents(
    documents=docs,
    collection_name="rag-sbert",
    embedding=embeddings,
    persist_directory = './db/chromadb',
)
print(vector_store.get())