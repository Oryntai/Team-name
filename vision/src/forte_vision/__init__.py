"""Локальный VLM-наблюдатель: видео кассы -> Observation для decision/."""

from .contract import ContractError, validate_observation
from .observer import PROFILES, analyze_video, failed_observation

__all__ = ["ContractError", "PROFILES", "analyze_video", "failed_observation", "validate_observation"]
