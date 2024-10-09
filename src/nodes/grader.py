from chains.retrieval_grader_chain import retrieval_grader


def grade_documents(state: dict) -> dict:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    for d in documents:
        grade = retrieval_grader.invoke(
            {"question": question, "documents": d.page_content}
        )["score"]
        if grade == "yes":
            filtered_docs.append(d)
    if len(filtered_docs)==0:
        search = "Yes"
    new_state = {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }
    return new_state
