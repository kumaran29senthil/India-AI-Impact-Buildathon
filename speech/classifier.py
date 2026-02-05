def classify_voice(features: dict):
    pitch_var = features["pitch_variance"]
    rms_var = features["rms_variance"]
    zcr = features["zcr_mean"]

    score = 0.0

    if pitch_var > 50:
        score += 0.4
    if rms_var > 0.01:
        score += 0.4
    if zcr > 0.05:
        score += 0.2

    score = round(score, 2)

    classification = "HUMAN" if score >= 0.5 else "AI_GENERATED"

    human_probability = score
    ai_probability = round(1 - score, 2)

    return {
        "classification": classification,
        "confidenceScore": score,
        "humanProbability": human_probability,
        "aiProbability": ai_probability,
        "explanation": (
            "Voice shows natural pitch, energy, and temporal variations"
            if classification == "HUMAN"
            else "Voice lacks natural human pitch and energy variation patterns"
        )
    }
