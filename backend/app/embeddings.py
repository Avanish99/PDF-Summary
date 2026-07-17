from urllib import response

from openai import OpenAI
import cohere

from app.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)
        self.client = OpenAI(api_key=settings.groq_api_key)

    def embed_text(self, text: str) -> list[float]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text cannot be empty.")
        response = self.cohere_client.embed(
            model=self.settings.cohere_embedding_model,
            texts=[cleaned_text],
            input_type="search_query",
            embedding_types=["float"],
    )

        return response.embeddings.float[0]



    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [text.strip() for text in texts if text.strip()]

        if not cleaned_texts:
            raise ValueError("Texts cannot be empty.")
        response = self.cohere_client.embed(
        model=self.settings.cohere_embedding_model,
        texts=cleaned_texts,
        input_type="search_document",
        embedding_types=["float"],
    )



        return response.embeddings.float