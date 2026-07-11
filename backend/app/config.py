from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-large"
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
