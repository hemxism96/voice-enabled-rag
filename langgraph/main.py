import os
from dotenv import load_dotenv

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


from typing import List

from typing_extensions import TypedDict

from nodes import RAG
import uuid

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
os.environ['CURL_CA_BUNDLE'] = ''


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


def langgragh():
    workflow = StateGraph(GraphState)
    rag = RAG()

    # Define the nodes
    workflow.add_node("retrieve", rag.retrieve)  # retrieve
    workflow.add_node("grade_documents", rag.grade_documents)  # grade documents
    workflow.add_node("generate", rag.generate)  # generatae
    workflow.add_node("web_search", rag.web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        rag.decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile()

    example = {"input": "List the members of the board directors in 2022."}
    response = custom_graph.invoke(
        {"question": example["input"], "steps": []}
    )
    print(response["generation"])



if __name__ == "__main__":
    langgragh()