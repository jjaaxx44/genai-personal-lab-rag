import hashlib
from dataclasses import dataclass

# PyMuPDF is AGPL-3.0 (or a paid Artifex commercial licence), and it is the one
# copyleft dependency in the stack. Every demo reaches PDFs through this module,
# so this import is the single place the obligation enters the app: reuse this
# code in anything closed-source -- including a service you only host yourself,
# because the AGPL's network clause counts that as conveying it -- and the whole
# application has to ship its source under the AGPL. See LICENSE.
import pymupdf

from .config import get_settings


class PdfValidationError(Exception):
    pass


@dataclass
class ParsedPdf:
    pages: list[str]
    page_count: int


def compute_doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf(filename: str, data: bytes) -> None:
    settings = get_settings()

    if not filename.lower().endswith(".pdf"):
        raise PdfValidationError("Only PDF files are accepted.")
    if not data.startswith(b"%PDF-"):
        raise PdfValidationError("File does not look like a valid PDF.")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise PdfValidationError(
            f"File is {size_mb:.1f} MB, over the {settings.max_upload_mb} MB limit."
        )

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.is_encrypted:
            raise PdfValidationError("Encrypted PDFs are not supported.")
        if doc.page_count > settings.max_pdf_pages:
            raise PdfValidationError(
                f"PDF has {doc.page_count} pages, over the {settings.max_pdf_pages} page limit."
            )


def extract_text(data: bytes) -> ParsedPdf:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
        return ParsedPdf(pages=pages, page_count=doc.page_count)
