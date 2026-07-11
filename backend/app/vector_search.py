from pinecone import Pinecone

from app.chunking import PDFChunk
from app.config import Settings


class VectorStoreService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.client.Index(settings.pinecone_index_name)

    def upsert_chunks(
        self,
        chunks: list[PDFChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        if not embeddings:
            raise ValueError("Embeddings cannot be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")

        vectors = []

        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        "document_id": chunk.document_id,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                }
            )

        self.index.upsert(vectors=vectors)

    def search(
        self,
        document_id: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={
                "document_id": {"$eq": document_id},
            },
        )

        matches = []

        for match in results.matches:
            matches.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "document_id": match.metadata.get("document_id"),
                    "page_number": match.metadata.get("page_number"),
                    "chunk_index": match.metadata.get("chunk_index"),
                    "text": match.metadata.get("text"),
                }
            )

        return matches

    def delete_document(self, document_id: str) -> None:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        self.index.delete(
            filter={
                "document_id": {"$eq": document_id},
            }
        )