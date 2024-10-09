import argparse
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def get_retriever(
        load_db: bool,
        document_path: Optional[str] = "./data",
        collection_name: Optional[str] = "rag-db",
        db_path: Optional[str] = "./db/chromadb",
        embedding_model_name: Optional[str] = "sentence-transformers/all-MiniLM-l6-v2"
    ) -> VectorStoreRetriever:
    """
    Creates a retriever for accessing documents
    either from a Chroma database or a specified directory.

    This function initializes embeddings using a specified model and either loads an existing 
    Chroma database or loads documents from a directory to create a new Chroma database. 
    It then returns a retriever object for querying.

    Args:
        load_db (bool): Flag indicating whether to load from an existing database (True) or 
                        load documents from a directory (False).
        document_path (Optional[str]): Path to the directory containing PDF documents. 
                                    Defaults to "./data".
        collection_name (Optional[str]): Name of the collection to use in the Chroma database. 
                                        Defaults to "rag-db".
        db_path (Optional[str]): Path to the directory where the Chroma database is stored. 
                                Defaults to "./db/chromadb".
        embedding_model_name (Optional[str]): Name of the Hugging Face model to use for embeddings. 
                                            Defaults to "sentence-transformers/all-MiniLM-l6-v2".

    Returns:
        VectorStoreRetriever: A retriever object that can be used to query the document store.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    if load_db:
        print("=====Load DB====")
        persistent_client = chromadb.PersistentClient(path=db_path)
        db = Chroma(
            client=persistent_client,
            embedding_function=embeddings,
            collection_name = collection_name
        )
    else:
        print("=====Load Documents====")
        loader = PyPDFDirectoryLoader(document_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=50
        )
        docs = text_splitter.split_documents(data)
        db = Chroma.from_documents(
            documents=docs,
            collection_name=collection_name,
            embedding=embeddings,
            persist_directory=db_path,
        )
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10}
    )
    return retriever


retriever = get_retriever(load_db=False)