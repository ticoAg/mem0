from openai import OpenAI

from mem0.embeddings.base import EmbeddingBase, EmbeddingModelConfig


class OpenAIEmbedding(EmbeddingBase):
    def __init__(self, config: EmbeddingModelConfig):
        self.client = OpenAI()
        self.model = config.model
        self.dims = config.dims

    def embed(self, text):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )
