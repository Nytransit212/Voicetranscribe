"""
Diarization Provider System

This package provides a modular diarization provider system that supports
multiple external APIs for speaker diarization, with fallback capabilities.
"""

from .base import DiarizationProvider, DiarizationResult, DiarizationError, ProviderStatus
from .assemblyai import AssemblyAIDiarizationProvider

__all__ = [
    'DiarizationProvider',
    'DiarizationResult', 
    'DiarizationError',
    'ProviderStatus',
    'AssemblyAIDiarizationProvider'
]