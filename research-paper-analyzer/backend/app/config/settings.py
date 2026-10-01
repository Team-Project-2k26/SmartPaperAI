"""
Application settings loaded from environment variables / .env file.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configurable settings for the backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str = "sqlite:///./research_papers.db"

    # Upload
    max_upload_size_mb: int = 50
    upload_dir: str = "uploads"

    # AI models
    use_abstractive_summary: bool = True
    summarizer_model: str = "sshleifer/distilbart-cnn-12-6"
    qa_model: str = "deepset/roberta-base-squad2"
    embedding_model: str = "all-MiniLM-L6-v2"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # Logging
    log_level: str = "INFO"


settings = Settings()
