from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from speech.whisper_service import WhisperService
from speech.feature_extractor import extract_audio_features
from speech.classifier import classify_voice
from schemas import VoiceDetectionRequest, VoiceDetectionResponse
import base64
import io
import librosa
import tempfile
import os
import logging
logging.basicConfig(level=logging.INFO)


# Initialize Whisper once (important for performance)
whisper_service = WhisperService(model_size="base")

app = FastAPI(
    title="AI Generated Voice Detection API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key (change before final submission)
API_KEY = "sk_test_123456789"

@app.get("/")
def read_root():
    return {"message": "AI Generated Voice Detection API is running. Use /api/voice-detection for requests."}

SUPPORTED_LANGUAGES = {
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
}


@app.post("/api/voice-detection",response_model=VoiceDetectionResponse)
def voice_detection(payload: VoiceDetectionRequest,x_api_key: str = Header(None)):

    # 1. API key validation
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or missing API key"
        )

    # 2. Validate request body
    language = payload.language
    audio_format = payload.audioFormat
    audio_base64 = payload.audioBase64


    if not language or not audio_format or not audio_base64:
        raise HTTPException(
            status_code=400,
            detail="Malformed request"
        )

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language"
        )

    if audio_format.lower() != "mp3":
        raise HTTPException(
            status_code=400,
            detail="Only MP3 audio format is supported"
        )

    # 3. Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(audio_base64)

        # Validate audio using librosa (do NOT modify audio)
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=None, mono=True)

        if y is None or len(y) == 0:
            raise ValueError("Empty audio")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted audio data"
        )

    # 4. Save temp audio file (needed for Whisper + feature extraction)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        # 5. Whisper transcription
        transcription = whisper_service.transcribe_bytes(audio_bytes)

        # 6. Feature extraction + classification
        features = extract_audio_features(temp_path)
        logging.info(f"Pitch variance: {features['pitch_variance']}")
        logging.info(f"RMS variance: {features['rms_variance']}")
        logging.info(f"ZCR mean: {features['zcr_mean']}")
        
        classification = classify_voice(features)


    finally:
        # Always clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 7. Final response
    return {
    "status": "success",
    "requestedLanguage": language,
    "detectedLanguage": transcription["language"],
    "classification": classification["classification"],
    "confidenceScore": classification["confidenceScore"],
    "humanProbability": classification["humanProbability"],
    "aiProbability": classification["aiProbability"],
    "explanation": classification["explanation"],
    "transcript": transcription["text"]
}




