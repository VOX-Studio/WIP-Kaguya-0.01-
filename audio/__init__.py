"""
Package audio pour Kaguya
Contient le pipeline STT/TTS et la gestion audio
"""

from .pipeline import (
    AudioPipeline,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
    AudioRecorder
)

__all__ = [
    'AudioPipeline',
    'SpeechToText',
    'TextToSpeech',
    'WakeWordDetector',
    'AudioRecorder'
]

__version__ = '0.1.0'