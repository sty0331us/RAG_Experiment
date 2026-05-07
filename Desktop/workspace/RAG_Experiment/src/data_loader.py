from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# Korean-aware separators (added '。' '、' for CJK punctuation)
_KO_SEPARATORS = ["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""]


def load_pdfs(data_dir: Path) -> List[Dict[str, Any]]:
    """Read all PDFs in *data_dir* and return raw document dicts."""
    data_dir = Path(data_dir)
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return []

    documents: List[Dict[str, Any]] = []
    for pdf_path in pdf_files:
        logger.info(f"Loading PDF: {pdf_path.name}")
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)

            full_text = "\n".join(pages_text)
            documents.append({
                "text": full_text,
                "metadata": {
                    "source": pdf_path.name,
                    "num_pages": len(reader.pages),
                },
            })
            logger.info(f"  → {len(reader.pages)} pages, {len(full_text):,} chars")
        except Exception as exc:
            logger.error(f"Failed to load {pdf_path.name}: {exc}")

    return documents


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict[str, Any]]:
    """Split documents into overlapping chunks using a smart text splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_KO_SEPARATORS,
        length_function=len,
    )

    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        raw_chunks = splitter.split_text(doc["text"])
        for idx, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": {
                        **doc["metadata"],
                        "chunk_idx": idx,
                    },
                })

    logger.info(
        f"Created {len(chunks)} chunks from {len(documents)} documents "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
