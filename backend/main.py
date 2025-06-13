# main.py - Vocalis Backend Server using Dialogflow STT/TTS and OpenAI LLM

import logging
import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from services.transcriber import DialogflowTranscriber
from services.tts import DialogflowTTS
from services.llm import OpenAILLM
from routes.websocket import websocket_endpoint
import config  # Assuming local config.py file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global service instances
transcription_service = None
llm_service = None
tts_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_service, llm_service, tts_service

    cfg = config.get_config()
    logger.info("Initializing services...")

    transcription_service = DialogflowTranscriber(
        project_id=cfg["dialogflow_project_id"],
        session_id="vocalis-session-001",
        language_code="en-US",
        credentials_path=cfg.get("google_credentials_path")
    )

    logger.info("All services initialized successfully")
    yield
    logger.info("Shutting down services... Shutdown complete")

app = FastAPI(
    title="SuaraSemar Backend",
    description="Speech-to-Speech AI Assistant Backend",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_transcription_service():
    return transcription_service

# def get_llm_service():
#     return llm_service

# def get_tts_service():
#     return tts_service

@app.get("/")
async def root():
    return {"status": "ok", "message": "SuaraSemar backend is running"}

@app.get("/health")
async def health_check():
    return {    
        "status": "ok",
        "services": {
            "transcription": transcription_service is not None,
            "llm": llm_service is not None,
            "tts": tts_service is not None
        },
        "config": {
            "dialogflow_project_id": config.DIALOGFLOW_PROJECT_ID
        }
    }

@app.get("/config")
async def get_full_config():
    if not all([transcription_service, llm_service, tts_service]):
        raise HTTPException(status_code=503, detail="Services not initialized")

    return {
        # "transcription": transcription_service.language_code,
        # "llm": llm_service.model,
        # "tts": tts_service.language_code,
        "system": config.get_config()
    }

# @app.websocket("/ws")
# async def websocket_route(websocket: WebSocket):
#     await websocket_endpoint(websocket, transcription_service, llm_service, tts_service)

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host=config.WEBSOCKET_HOST,
#         port=config.WEBSOCKET_PORT,
#         reload=True
#     )
