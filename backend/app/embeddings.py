from openai import OpenAI

from app.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed_text(self, text: str) -> list[float]:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Text cannot be empty.")

        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=cleaned_text,
        )

        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [text.strip() for text in texts if text.strip()]

        if not cleaned_texts:
            raise ValueError("Texts cannot be empty.")

        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=cleaned_texts,
        )

        return [item.embedding for item in response.data]