import logging
import os
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile
from openai import OpenAI

from app.config import Settings
import uuid

from app.chunking import chunk_pdf_pages
from app.embeddings import EmbeddingService
from app.pdf_loader import load_pages_from_pdf
from app.vector_search import VectorStoreService



class PdfRagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
    api_key=settings.groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)
        self.embedding_service = EmbeddingService(settings)
        self.vector_store = VectorStoreService(settings)

    def index_pdf(self, upload: UploadFile) -> str:
        suffix = Path(upload.filename or "document.pdf").suffix or ".pdf"
        temporary_path: str | None = None
    

        document_id = str(uuid.uuid4())

        logger = logging.getLogger(__name__)


        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary_path = temporary.name
                while chunk := upload.file.read(1024 * 1024):
                    temporary.write(chunk)
                
            loaded_pages = load_pages_from_pdf(temporary_path)
            chunks = chunk_pdf_pages(
                document_id=document_id,
                pages=loaded_pages,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,

            )
            embeddings = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
            self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

            return document_id
        except Exception as e:
            logger.exception("Document processing failed")
            try:
                self.vector_store.delete_document(document_id)
            except Exception:
                logger.exception("Failed to cleanup vector store")
            raise HTTPException(
        status_code=500,
        detail="Unable to process document. Please try again later."
    )
        finally:
            upload.file.close()
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    def summarize(self, document_id: str) -> str:
        matches = self.vector_store.search(
            document_id=document_id,
            query_embedding=self.embedding_service.embed_text("Summarize the document."),
            top_k=20,
        )
        if not matches:
            return "I could not find that in the document. Hence Could not summarize the document."
        context="\n\n".join(
            [
                f"[Page {match['page_number']}, Chunk {match['chunk_index']}]\n{match['text']}"
                for match in matches
            ]           
        )
        response = self.client.responses.create(
            model=self.settings.groq_model,
            instructions=(
                "You are a precise document summarizer. Use only information retrieved "
                "from the uploaded PDF. If information is unavailable, say so. Return "
                "clear Markdown and do not invent facts."
            ),
            input=(
                f"Document context:\n{context}\n\n"
                "Summarize this PDF using these sections:\n"
                "1. Overview\n"
                "2. Key points\n"
                "3. Important facts or evidence\n"
                "4. Conclusions\n"
                "5. Action items, if any"
            ),
           
        )
        return response.output_text

    def ask(self, document_id: str, question: str) -> str:
        embedding = self.embedding_service.embed_text(question)
        matches = self.vector_store.search(
            document_id=document_id,
            query_embedding=embedding,
            top_k=10,
        )
        if not matches:
            return "I could not find that in the document."
        
        context = "\n\n".join(
        [
            f"[Page {match['page_number']}, Chunk {match['chunk_index']}]\n{match['text']}"
            for match in matches
        ]
    )
        response = self.client.responses.create(
            model=self.settings.groq_model,
            instructions=(
               "You are a careful research assistant. Answer using only information retrieved "
                "from the uploaded PDF. For in-depth questions, synthesize across all relevant "
                "retrieved sections, compare details when useful, and explain reasoning step by step. "
                'If the answer is absent, say "I could not find that in the document." '
                "Do not use outside knowledge or invent details. Return clear Markdown."
            ),
            input=(
            f"Document context:\n{context}\n\n"
            f"Question:\n{question}"
        ),
            
        )
        return response.output_text

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
