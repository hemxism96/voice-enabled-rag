from typing import List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]



class Graph:
    @staticmethod
    def create(nodes):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", nodes.retrieve)  # retrieve
        workflow.add_node("grade_documents", nodes.grade_documents)  # grade documents
        workflow.add_node("generate", nodes.generate)  # generatae
        workflow.add_node("web_search", nodes.web_search)  # web search

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            nodes.decide_to_generate,
            {
                "search": "web_search",
                "generate": "generate",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        custom_graph = workflow.compile()
        return custom_graph