import os
from dotenv import load_dotenv
import uuid

from workflow.graph import Graph
from nodes.vector_store import get_retriever
from nodes.reranker import get_reranker
from nodes.grader import get_grader_chain
from nodes.web_search import get_web_tool
from nodes.llm import get_generator_chain
from nodes.base import Nodes
from module.stt import SpeechToTextConverter

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


class NodeHelpers:
    """Helper class to initialize nodes."""
    retriever = get_retriever(load_db=True, collection_name="rag-sbert")
    reranker = get_reranker()
    retrieval_grader = get_grader_chain()
    web_search_tool = get_web_tool()
    generator = get_generator_chain(llm_repo_id=LLM_REPO_ID, hf_api_key=HF_API_KEY)


def main() -> None:
    """
    Main function to initialize the workflow and process user queries.

    This function sets up the necessary components for the application, including the 
    speech-to-text converter, the nodes for managing different stages of the workflow, 
    and the graph that defines the flow of execution. It then captures a query from the 
    user, invokes the workflow with the query, and prints the generated response.

    Steps:
        1. Initialize nodes with NodeHelpers.
        2. Create a workflow graph using the nodes.
        3. Convert speech input to text using the SpeechToTextConverter.
        4. Invoke the workflow with the user's query and print the generated response.
    """
    nodes = Nodes(helpers=NodeHelpers())
    custom_graph = Graph.create(nodes)
    stt = SpeechToTextConverter(whisper_model="base")
    query = stt()
    print(f"Question: {query}")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = custom_graph.invoke({"question": query, "steps": []}, config)
    res = {"response": state_dict["generation"], "steps": state_dict["steps"]}
    print(f"Steps: {res['steps']}")
    print(f"Response: {res['response']}")


if __name__ == "__main__":
    main()
