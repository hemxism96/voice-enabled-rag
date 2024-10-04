from typing import Optional

import speech_recognition as sr


class SpeechToTextConverter:
    """
    A class to convert speech into text using a specified Whisper model.

    Attributes:
        whisper_model (str): The name of the Whisper model to use for recognition.
        language (Optional[str]): The language to be used for recognition, defaults to "english".
    """
    def __init__(self, whisper_model: str, language: Optional[str] = "english") -> None:
        """
        Initializes the SpeechToTextConverter

        Args:
            whisper_model (str): The name of the Whisper model to use.
            language (Optional[str]): The language for speech recognition (default is "english").
        """
        self.recognizer = sr.Recognizer()
        self.whisper_model = whisper_model
        self.language = language


    def listen(self) -> sr.AudioData:
        """
        Listens to audio from the microphone and returns the audio data.

        Returns:
            sr.AudioData: The audio data captured from the microphone.
        """
        with sr.Microphone() as source:
            print("Ask something!")
            audio = self.recognizer.listen(source)
        return audio


    def convert(self, audio: sr.AudioData) -> str:
        """
        Converts the provided audio data to text using the Whisper model.

        Args:
            audio (sr.AudioData): The audio data to convert to text.

        Returns:
            str: The transcribed text from the audio.

        Raises:
            Exception: If the audio could not be processed by the Whisper model.
        """
        try:
            text = self.recognizer.recognize_whisper(
                audio_data=audio,
                model=self.whisper_model,
                language=self.language
            )
            return text
        except Exception as e:
            print(f"Could not request results from Whisper.\n{e}")
            return ""


    def __call__(self) -> str:
        """
        Listens to audio and converts it to text.

        Returns:
            str: The transcribed text from the audio.
        """
        audio = self.listen()
        text = self.convert(audio)
        return text
