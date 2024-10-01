import os
from dotenv import load_dotenv
import argparse

from langchain_huggingface import HuggingFaceEndpoint

from vector_store import set_vector_store
from stt import SpeechToTextConverter
from llm import getLanggraph

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

from llm import retrieval_grader_prompt, generator_prompt
from typing_extensions import TypedDict
from typing import List

from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

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


def main(llm_model_name: str, embedding_model_name: str, documents_path: str) -> None:
    # Creating database form documents
    try:
        db = set_vector_store(
            hf_api_key = HF_API_KEY,
            embedding_model_name = embedding_model_name, 
            collection_name = "rag-rd"
        )
    except FileNotFoundError as e:
        print(e)

    # Get voice query qnd convert to text
    stt = SpeechToTextConverter(
        whisper_model="base", 
    )
    #query = stt()
    #print("Question: ", query)
    
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=HF_API_KEY,
    )
    retriever = db.as_retriever(k=3)
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
    web_search_tool = TavilySearchResults(k=3)
    rag_chain = generator_prompt | llm | StrOutputParser()
    
    workflow = StateGraph(GraphState)

    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]
        print(question)
        documents = retriever.invoke(question)
        steps = state["steps"]
        steps.append("retrieve_documents")
        return {"documents": documents, "question": question, "steps": steps}


    def grade_documents(state):
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
        return {
            "documents": filtered_docs,
            "question": question,
            "search": search,
            "steps": steps,
        }


    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        search = state["search"]
        if search == "Yes":
            return "search"
        else:
            return "generate"


    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        question = state["question"]
        documents = state.get("documents", [])
        steps = state["steps"]
        steps.append("web_search")
        web_results = web_search_tool.invoke({"query": question})
        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]
        )
        return {"documents": documents, "question": question, "steps": steps}


    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"documents": documents, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }


    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("web_search", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
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
    query = "List the members of the board directors in 2022."
    response = custom_graph.invoke(
        {"question": query, "steps": []}
    )
    print(response)




def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument(
        "-m",
        "--model",
        default="mistral",
        help="The name of the LLM model to use.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default="nomic-embed-text",
        help="The name of the embedding model to use.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="Research",
        help="The path to the directory containing documents to load.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.embedding_model, args.path)