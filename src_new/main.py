
import os
from dotenv import load_dotenv

from workflow.graph import Graph
from nodes.vector_store import get_retriever
from nodes.grader import get_grader_chain
from nodes.web_search import get_web_tool
from nodes.llm import get_generator_chain
from nodes.base import Nodes
from module.stt import SpeechToTextConverter

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")


class NodeHelpers:
    retriever = get_retriever()
    retrieval_grader = get_grader_chain()
    web_search_tool = get_web_tool()
    generator = get_generator_chain()


def main(llm_repo_id: str, embedding_model_name: str, documents_path: str) -> None:
    nodes = Nodes(helpers=NodeHelpers())
    custom_graph = Graph.create(nodes)
    stt = SpeechToTextConverter(whisper_model="base")

    query = stt()
    #query = "List the members of the board directors in 2022."
    print("Question: ", query)
    response = custom_graph.invoke(
        {"question": query, "steps": []}
    )
    print(response["generation"])


if __name__ == "__main__":
    main("meta-llama/Meta-Llama-3-8B-Instruct", args.embedding_model, args.path)
