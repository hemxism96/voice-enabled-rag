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
    def create(nodes) -> StateGraph:
        """
        Creates a state graph workflow using the provided nodes.

        This method initializes a StateGraph and adds nodes representing different actions 
        such as rephrasing, retrieving documents, grading them, generating responses, 
        and performing web searches. It also defines the edges that dictate the flow of 
        the workflow.

        Args:
            nodes: A Nodes object containing methods corresponding to various workflow steps, 
                including rephrase, retrieve, grade_documents, generate, and web_search.

        Returns:
            StateGraph: A compiled StateGraph object that represents the workflow.
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", nodes.retrieve)
        workflow.add_node("rerank", nodes.rerank)
        workflow.add_node("grade_documents", nodes.grade_documents)
        workflow.add_node("generate", nodes.generate)
        workflow.add_node("web_search", nodes.web_search)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "grade_documents")
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
