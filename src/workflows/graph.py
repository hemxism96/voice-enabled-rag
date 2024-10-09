from typing import List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from nodes.retriever import retrieve
from nodes.reranker import rerank
from nodes.grader import grade_documents
from nodes.generator import generate
from nodes.web_tool import web_search
from nodes.decision_node import decide_to_generate


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


def create_workflow() -> CompiledStateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search", web_search)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile()
    return custom_graph
