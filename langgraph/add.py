__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFDirectoryLoader("./data")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
docs = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
vector_store = Chroma.from_documents(
    documents=docs,
    collection_name="rag-rd",
    embedding=embeddings,
    persist_directory = './db/chromadb',
)
print(vector_store.get())