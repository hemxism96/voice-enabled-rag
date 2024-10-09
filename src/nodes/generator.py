from chains.generator_chain import generator_chain


def generate(state: dict) -> dict:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = "\n\n".join(doc.page_content for doc in state["documents"])
    generation = generator_chain.invoke({"documents": documents, "question": question})
    steps = state["steps"]
    steps.append("generate_answer")
    new_state = {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }
    return new_state
