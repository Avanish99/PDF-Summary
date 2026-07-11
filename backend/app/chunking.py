from dataclasses import dataclass



from app.pdf_loader import PDFDocument


@dataclass
class PDFChunk:
    document_id: str
    chunk_id: str
    page_number: int
    chunk_index: int
    text: str

def chunk_pdf_pages(
    document_id: str,
    pages: list[PDFDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    
) -> list[PDFChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    
    chunks: list[PDFChunk] = []
    chunk_index = 0
    step_size = chunk_size - chunk_overlap

    for page in pages:
        text = page.text.strip()
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    PDFChunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}-{chunk_index}",
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                    )
                )
                chunk_index += 1

            start += step_size

    if not chunks:
        raise ValueError("No chunks were created from this PDF.")

    return chunks

