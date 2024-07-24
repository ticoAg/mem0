from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class EmbeddingBase(ABC):
    @abstractmethod
    def embed(self, text):
        """
        Get the embedding for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        pass


class EmbeddingModelConfig(BaseModel):
    model: str = Field(
        description="Name of the embedding model (e.g., 'text-embedding-3-small')",
        default="text-embedding-3-small",
    )
    dims: int = Field(
        description="Dimensions of the embedding model",
        default=1536,
    )
