from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    cohere_api_key: str
    cohere_embedding_model: str = "embed-english-v3.0"
    pinecone_api_key: str
    pinecone_index_name: str
    max_upload_mb: int = 25
    chunk_size: int = 3500
    chunk_overlap: int = 600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )




@lru_cache
def get_settings() -> Settings:
    return Settings()
