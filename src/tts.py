from TTS.api import TTS


class TextToSpeechConverter:
    """AudioRecorder class for recording voice from mic
    """
    def __init__(self, whisper_model, language="english"):
        self.device = "cpu"
        self.tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(self.device)

    def convert(self, llm_answer):
        try:
            self.tts.tts_to_file(text=llm_answer, speaker_wav="./audio/tmp.wav", language="en", file_path="./audio/tmp.wav")
        except Exception as e:
            print(f"Could not convert a text to audio by Coqui; {e}")
