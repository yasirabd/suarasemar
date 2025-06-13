import numpy as np
from scipy.io import wavfile
from backend.services.transcriber import DialogflowTranscriber 

# Load WAV file as numpy array
def load_audio(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

# Instantiate the transcriber
transcriber = DialogflowTranscriber(
    project_id="robosemarhealth-jxgh",
    session_id="test-session",
    language_code="id",
    credentials_path="E:\Sanditheta Solusi Digital\suarasemar\credentials\\robosemarhealth-jxgh-8a8b99fefe05.json"
)

# Load audio
audio_data = load_audio("E:\Sanditheta Solusi Digital\suarasemar\\backend\\tests\\test.wav")

# Transcribe
text, meta = transcriber.transcribe(audio_data)

# Output results
print("Transcription:", text)
print("Metadata:", meta)
