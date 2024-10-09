import gradio as gr
from transformers import pipeline

from workflows.graph import create_workflow

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
custom_graph = create_workflow()


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
