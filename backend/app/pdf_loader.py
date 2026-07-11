import fitz  # PyMuPDF

from dataclasses import dataclass

@dataclass
class PDFDocument:
    page_number: int
    text: str

def load_pages_from_pdf(file_path: str) -> list[PDFDocument]:
    pages: list[PDFDocument] = []

    with fitz.open(file_path) as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    PDFDocument(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF. It may be scanned or image-based."
        )

    return pages
