__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
import chromadb

persistent_client = chromadb.PersistentClient(path="./db/chromadb")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
vector_store = Chroma(
    client=persistent_client,
    embedding_function=embeddings,
    collection_name = "rag-rd"
)
results = vector_store.similarity_search(
    "List the members of the board directors in 2022."
)
print(results)
print(results[0].page_content)