from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError

from app.config import Settings, get_settings
from app.rag import PdfRagService
from app.schemas import (
    AnswerResponse,
    DocumentResponse,
    HealthResponse,
    QuestionRequest,
    SummaryRequest,
)

app = FastAPI(
    title="PDF RAG API",
    description="Upload PDFs, generate summaries, and ask grounded questions.",
    version="1.0.0",
)

# Restrict this list to the real frontend URL before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173","https://pdf-summary-sable.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_rag_service() -> PdfRagService:
    return PdfRagService(get_settings())


def openai_error(error: APIError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"OpenAI request failed: {error.message}",
    )


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", model=settings.openai_model)


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    rag: PdfRagService = Depends(get_rag_service),
) -> DocumentResponse:
    filename = file.filename or "document.pdf"
    is_pdf = file.content_type == "application/pdf" or filename.lower().endswith(".pdf")

    if not is_pdf:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF must be {settings.max_upload_mb} MB or smaller.",
        )

    try:
        document_id = rag.index_pdf(file)
    except APIError as error:
        raise openai_error(error) from error

    return DocumentResponse(document_id=document_id, filename=filename)


@app.post("/summary", response_model=AnswerResponse)
def summarize_document(
    request: SummaryRequest,
    rag: PdfRagService = Depends(get_rag_service),
) -> AnswerResponse:
    try:
        return AnswerResponse(answer=rag.summarize(request.document_id))
    except APIError as error:
        raise openai_error(error) from error


@app.post("/ask", response_model=AnswerResponse)
def ask_document(
    request: QuestionRequest,
    rag: PdfRagService = Depends(get_rag_service),
) -> AnswerResponse:
    try:
        return AnswerResponse(
            answer=rag.ask(request.document_id, request.question.strip())
        )
    except APIError as error:
        raise openai_error(error) from error


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    rag: PdfRagService = Depends(get_rag_service),
) -> None:
    try:
        rag.delete_document(document_id)
    except APIError as error:
        raise openai_error(error) from error
