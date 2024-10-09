def decide_to_generate(state: dict) -> str:
    """
    Determines whether to generate an answer, or do web research.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    search = state["search"]
    if search == "Yes":
        next_node = "search"
    else:
        next_node = "generate"
    return next_node
