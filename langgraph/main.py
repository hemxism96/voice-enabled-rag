import os
from dotenv import load_dotenv

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOllama


from typing import List

from typing_extensions import TypedDict

from nodes import RAG
import uuid

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


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
    rag = RAG(input_path="./data")

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
    return custom_graph


def main():
    custom_graph = langgragh()
    def predict_custom_agent_local_answer(example: dict):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state_dict = custom_graph.invoke(
            {"question": example["input"], "steps": []}, config
        )
        return {"response": state_dict["generation"], "steps": state_dict["steps"]}


    example = {"input": "List the members of the board directors in 2022."}
    response = predict_custom_agent_local_answer(example)
    print(response)



if __name__ == "__main__":
    main()