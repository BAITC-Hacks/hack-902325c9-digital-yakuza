from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str
    app_env: Literal["development", "test", "production"]
    debug: bool
    database_url: SecretStr
    cors_origins: list[str]
    openai_api_key: SecretStr = SecretStr("")
    beeline_agent_dir: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
