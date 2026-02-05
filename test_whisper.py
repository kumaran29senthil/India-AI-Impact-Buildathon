import whisper

print("Loading Whisper model...")
model = whisper.load_model("base")

print("Transcribing audio...")
result = model.transcribe("sample.mp3")

print("\n--- TRANSCRIPTION RESULT ---")
print("Detected Language:", result["language"])
print("Text:", result["text"])
