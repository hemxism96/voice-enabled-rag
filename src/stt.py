import speech_recognition as sr


class SpeechToTextConverter:
    """AudioRecorder class for recording voice from mic
    """
    def __init__(self, whisper_model, language="english"):
        self.recognizer = sr.Recognizer()
        self.whisper_model = whisper_model
        self.language = language

    def listen(self):
        with sr.Microphone() as source:
            print("Say something!")
            audio = self.recognizer.listen(source)
        return audio
    
    def convert(self, audio):
        try:
            text = self.recognizer.recognize_whisper(
                audio, 
                model=self.whisper_model,
                language=self.language
            )
            return text
        except Exception as e:
            print(f"Could not request results from Whisper; {e}")

    def __call__(self):
        audio = self.listen()
        text = self.convert(audio)
        return text