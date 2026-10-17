"""Shared configuration; importing this module never imports an ML runtime."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORTE_", env_file=".env", env_file_encoding="utf-8", extra="ignore",
        hide_input_in_errors=True,
    )

    environment: Literal["development", "production"] = "development"
    api_key: SecretStr = SecretStr("")
    data_dir: Path = Path("data")
    deadline_seconds: float = Field(default=300, gt=0, le=300)
    max_upload_mb: int = Field(default=64, ge=29, le=1024)
    max_pending_jobs: int = Field(default=5, ge=1, le=100)
    profile: str = "standard"
    analyzer_factory: str = "backend.adapters.unavailable:create_analyzer"
    rules_factory: str = "backend.adapters.unavailable:create_rules"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    frontend_dist: Path = Path("frontend/dist")
    worker_poll_sec: float = Field(default=0.2, gt=0, le=5)
    max_attempts: int = Field(default=3, ge=1, le=10)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def plugins_configured(self) -> bool:
        return all(
            ":" in factory and not factory.startswith(("backend.adapters.", "backend.tests.", "tests."))
            for factory in (self.analyzer_factory, self.rules_factory)
        )

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        self.data_dir = self.data_dir.expanduser().resolve()
        self.frontend_dist = self.frontend_dist.expanduser().resolve()
        jury_key = self.api_key.get_secret_value()
        normalized_key = jury_key.casefold()
        if jury_key and normalized_key.startswith(("replace-with-", "your-", "change-me")):
            raise ValueError("FORTE_API_KEY must not use a placeholder value")
        if self.environment == "production":
            if not self.plugins_configured:
                raise ValueError("Production requires real analyzer and rules factories; development/test adapters are forbidden")
            if len(jury_key) < 32:
                raise ValueError("Production requires a generated FORTE_API_KEY with at least 32 characters")
        return self
