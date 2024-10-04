import os
from dotenv import load_dotenv

import gradio as gr
from transformers import pipeline

from workflow.graph import Graph
from nodes.base import Nodes
from nodes.vector_store import get_retriever
from nodes.reranker import get_reranker
from nodes.grader import get_grader_chain
from nodes.web_search import get_web_tool
from nodes.llm import get_generator_chain

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


class NodeHelpers:
    """
    Helper class to initialize nodes.
    Set 'load_db' to False if you haven't loaded documents in the database.
    The demo assumes a pre-initialized database with the 'rag-sbert' collection.
    """
    retriever = get_retriever(load_db=True, collection_name="rag-sbert")
    reranker = get_reranker()
    retrieval_grader = get_grader_chain()
    web_search_tool = get_web_tool()
    generator = get_generator_chain(llm_repo_id=LLM_REPO_ID, hf_api_key=HF_API_KEY)


nodes = Nodes(helpers=NodeHelpers())
custom_graph = Graph.create(nodes)


def transcription_speech(_audio: gr.Audio) -> str:
    """Transcribes speech from the given audio.

    Args:
        audio (gr.Audio): The audio data to convert to text.

    Returns:
        str: The transcribed text from the audio.
    """
    if _audio is None:
        gr.Warning("No Audio Found. Please Retry")
        output = ""
    output = transcriber(_audio)
    return output["text"]


def generate_response(_question: str) -> str:
    """Generates a response based on the provided question using the custom graph.

    Args:
        question (str): Given question.

    Returns:
        str: llm generated response.
    """
    state_dict = custom_graph.invoke({"question": _question, "steps": []})
    _response = state_dict["generation"]
    return _response


with gr.Blocks() as app:
    gr.Markdown("Voice-enabled conversational assistant")

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources="microphone", type="filepath", label="Ask a question!")
            question = gr.Textbox(label="Question")
            submit_button = gr.Button("Submit")
        with gr.Column():
            response = gr.Textbox(label="Response")

    audio.change(fn=transcription_speech, inputs=audio, outputs=question)
    submit_button.click(fn=generate_response, inputs=question, outputs=response)

app.launch()
