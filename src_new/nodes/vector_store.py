from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


def get_retriever(
        hf_api_key: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-l6-v2",
        document_path: str = None,
        collection_name: str = "rag-db",
        db_path: str = "./db/chromadb"
    ) -> Chroma:
    """
    Loads documents from the specified directory into the Chroma database
    after splitting the text into chunks.

    Returns:
        Chroma: The Chroma database with loaded documents.
    """
    # embeddings = HuggingFaceHubEmbeddings(
    #     repo_id=embedding_model_name, 
    #     huggingfacehub_api_token=hf_api_key
    # )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    if document_path:
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
    else:
        persistent_client = chromadb.PersistentClient(path=db_path)
        db = Chroma(
            client=persistent_client,
            embedding_function=embeddings,
            collection_name = collection_name
        )
    retriever = db.as_retriever()
    return retriever