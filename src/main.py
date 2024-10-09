import uuid

from workflows.graph import create_workflow
from modules.stt import SpeechToTextConverter


def main() -> None:
    """
    Main function to initialize the workflow and process user queries.

    This function sets up the necessary components for the application, including the 
    speech-to-text converter, the nodes for managing different stages of the workflow, 
    and the graph that defines the flow of execution. It then captures a query from the 
    user, invokes the workflow with the query, and prints the generated response.

    Steps:
        1. Create a workflow graph using the nodes.
        2. Convert speech input to text using the SpeechToTextConverter.
        3. Invoke the workflow with the user's query and print the generated response.
    """
    custom_graph = create_workflow()
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
