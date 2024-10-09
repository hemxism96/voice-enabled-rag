from langchain.schema import Document
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(model_name="mixedbread-ai/mxbai-rerank-large-v1")


def rerank(state: dict) -> dict:
    """
    Rerank documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains best retrieved documents
    """
    question = state["question"]
    documents = [d.page_content for d in state["documents"]]
    reranked_document = reranker.rank(
        question, documents, return_documents=True, top_k=3
    )
    reranked_document = [
        Document(page_content=d["text"], metadata={"source": "rerank_doc"})
        for d in reranked_document
    ]
    steps = state["steps"]
    steps.append("rerank")
    new_state = {"documents": reranked_document, "question": question, "steps": steps}
    return new_state
