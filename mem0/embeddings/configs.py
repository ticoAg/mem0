from typing import Optional

from pydantic import BaseModel, Field, field_validator

from mem0.embeddings.base import EmbeddingModelConfig


class EmbedderConfig(BaseModel):
    provider: str = Field(
        description="Provider of the embedding model (e.g., 'ollama', 'openai')",
        default="openai",
    )
    config: EmbeddingModelConfig = Field(
        description="Configuration for the specific embedding model", default=None
    )

    @field_validator("config")
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        if provider in ["openai", "ollama"]:
            return v
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
