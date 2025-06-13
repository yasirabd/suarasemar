import sounddevice as sd
from typing import Dict, Tuple, Union
import logging
import numpy as np
from config.components.config import ConfigurationManager



logger = logging.getLogger(__name__)

STREAM_REGISTRY = {}
CONFIG = ConfigurationManager()

def register_stream(engine: str):
    def decorator(cls):
        STREAM_REGISTRY[engine] = cls
        return cls

    return decorator

@register_stream("sounddevice")
class SounddeviceStream:
    def __init__(self):
        config = CONFIG["microphone"]
        self.channels: int = int(config["channels"])
        self.samplerate: int = int(config["samplerate"])
        self.device_index: int = int(config["device_index"])
        self.chunk: int = int(config["chunk"])
        self.dtype: str = str(config["dtype"])
        self.stream_type = config["sounddevice"]["stream_type"]
        self.stream_cls = None
        input_stream_registry = {"raw": sd.RawInputStream, "numpy": sd.InputStream}

        self.stream_cls = input_stream_registry.get(self.stream_type, None)

        if self.stream_cls == None:
            raise ValueError(f"Unknown stream type: {self.stream_type}")

        self.mic_audio = self.stream_cls
        self.stream = self.mic_audio

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.close()

    def stop(self):
        self.stream.stop()

    def terminate(self):
        self.stream.abort()

    def open(self):
        self.stream = self.mic_audio(
            channels=self.channels,
            samplerate=self.samplerate,
            device=self.device_index,
            blocksize=self.chunk,
            dtype=self.dtype,
        )        

    def read(self):
        return self.stream.read(self.chunk)
    
    def is_active(self):
        return self.stream.active