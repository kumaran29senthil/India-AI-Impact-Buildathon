import whisper
import tempfile
import os

class WhisperService:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)

    def transcribe_bytes(self, audio_bytes: bytes):
        # Whisper needs a file path, so we create a temp mp3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        try:
            result = self.model.transcribe(temp_audio_path)
            return {
                "language": result["language"],
                "text": result["text"]
            }
        finally:
            os.remove(temp_audio_path)
