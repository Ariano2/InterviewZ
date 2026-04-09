"""
file_parser.py
Parses uploaded PDF or DOCX files into plain text.

PyMuPDF >= 1.25  →  import fitz  (unchanged public API)
python-docx >= 1.1.0
"""

import os
import tempfile

import fitz  # PyMuPDF
from docx import Document


def parse_uploaded_file(uploaded_file) -> str:
    """
    Accept a Streamlit UploadedFile object.
    Returns extracted plain text string.
    Raises ValueError for unsupported types.
    """
    name = uploaded_file.name
    suffix = os.path.splitext(name)[-1].lower()

    if suffix not in (".pdf", ".docx", ".doc"):
        raise ValueError(f"Unsupported file type '{suffix}'. Please upload PDF or DOCX.")

    # Write to a temp file so libraries can open by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            return _extract_pdf(tmp_path)
        else:
            return _extract_docx(tmp_path)
    finally:
        os.unlink(tmp_path)


def _extract_pdf(path: str) -> str:
    """Extract text from all pages using PyMuPDF."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))  # "text" mode = plain text
    doc.close()
    return "\n\n".join(pages).strip()


def _extract_docx(path: str) -> str:
    """Extract paragraph text from DOCX."""
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()
