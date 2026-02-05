from pydantic import BaseModel, Field

class VoiceDetectionRequest(BaseModel):
    language: str = Field(example="English")
    audioFormat: str = Field(example="mp3")
    audioBase64: str

class VoiceDetectionResponse(BaseModel):
    status: str
    requestedLanguage: str
    detectedLanguage: str
    classification: str
    confidenceScore: float
    humanProbability: float
    aiProbability: float
    explanation: str
    transcript: str
