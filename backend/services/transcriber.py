# Speech-to-Text Transcription Service (Google Dialogflow Version)

import logging
import io
import numpy as np
from typing import Dict, Any, Tuple
from google.cloud import dialogflow_v2 as dialogflow
from google.cloud.dialogflow_v2.types import InputAudioConfig, QueryInput, AudioEncoding
import os
import time
import wave

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DialogflowTranscriber:
    """
    Speech-to-Text service using Google Dialogflow.

    This class handles transcription of speech audio using Dialogflow's detectIntent API.
    """

    def __init__(
        self,
        project_id: str,
        session_id: str,
        language_code: str = "en-US",
        credentials_path: str = None
    ):
        """
        Initialize the transcription service.

        Args:
            project_id: Google Cloud Dialogflow project ID
            session_id: Unique session ID for each conversation
            language_code: Language code for transcription (default is 'en-US')
            credentials_path: Path to Google credentials JSON (if not already set in env)
        """
        self.project_id = project_id
        self.session_id = session_id
        self.language_code = language_code

        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(project_id, session_id)

        logger.info(f"Initialized Dialogflow Transcriber with project_id={project_id}, session_id={session_id}")

    def _extract_sample_rate(self, audio: np.ndarray) -> int:
        """Extract the sample rate from the WAV header in the numpy audio array."""
        try:
            with io.BytesIO(audio) as byte_stream:
                with wave.open(byte_stream, 'rb') as wav_file:
                    return wav_file.getframerate()
        except Exception as e:
            logger.warning(f"Could not extract sample rate, defaulting to 16000 Hz: {e}")
            return 16000

    def transcribe(self, audio: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio using Google Dialogflow.

        Args:
            audio: Audio data as numpy array (uint8 with WAV headers)

        Returns:
            Tuple[str, Dict[str, Any]]: 
                - Transcribed text
                - Metadata dictionary
        """
        start_time = time.time()
        try:
            if audio.dtype != np.uint8:
                raise ValueError("Dialogflow requires WAV format audio with headers in uint8.")

            # Convert numpy array to bytes for Dialogflow input
            audio_bytes = io.BytesIO(audio).read()

            # Extract actual sample rate from audio header
            sample_rate = self._extract_sample_rate(audio)

            audio_config = InputAudioConfig(
                audio_encoding=AudioEncoding.AUDIO_ENCODING_LINEAR_16,
                language_code=self.language_code,
                sample_rate_hertz=sample_rate,
            )

            query_input = QueryInput(audio_config=audio_config)

            response = self.session_client.detect_intent(
                request={
                    "session": self.session,
                    "query_input": query_input,
                    "input_audio": audio_bytes
                }
            )

            query_result = response.query_result
            full_text = query_result.query_text
            processing_time = time.time() - start_time

            metadata = {
                "intent": query_result.intent.display_name,
                "confidence": query_result.intent_detection_confidence,
                "language": self.language_code,
                "processing_time": processing_time,
                "sample_rate_used": sample_rate
            }

            logger.info(f"Transcription completed in {processing_time:.2f}s: {full_text}")
            return full_text, metadata

        except Exception as e:
            logger.error(f"Dialogflow transcription error: {e}")
            return "", {"error": str(e)}
