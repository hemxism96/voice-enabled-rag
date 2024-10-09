from modules.vector_store import retriever


def retrieve(state: dict) -> dict:
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    new_state = {"documents": documents, "question": question, "steps": steps}
    return new_state
