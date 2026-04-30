"""Build or rebuild the local Chroma vector index from PDF documents."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

from src.config import get_embeddings, get_settings


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def find_pdf_files(data_dir: Path) -> list[Path]:
    """Return all PDFs under the configured data directory."""

    return sorted(path for path in data_dir.rglob("*.pdf") if path.is_file())


def load_pdfs(data_dir: Path) -> list[Document]:
    """Load PDF pages with stable source metadata."""

    documents: list[Document] = []
    pdf_files = find_pdf_files(data_dir)
    if not pdf_files:
        logger.warning("No PDF files found under %s.", data_dir)
        return documents

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = str(pdf_path.relative_to(data_dir.parent))
                page.metadata["file_name"] = pdf_path.name
            documents.extend(pages)
            logger.info("Loaded %s pages from %s.", len(pages), pdf_path.name)
        except Exception as exc:
            logger.exception("Failed to load %s: %s", pdf_path, exc)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into retrieval chunks."""

    settings = get_settings()
    # Character counts are used here to keep ingestion fully offline. A chunk
    # size of 3000-4000 chars is roughly in the requested 500-1000 token range.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(rebuild: bool = False) -> int:
    """Build the Chroma index and return the number of indexed chunks."""

    settings = get_settings()
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    if rebuild and settings.vectorstore_dir.exists():
        logger.info("Rebuilding vectorstore at %s.", settings.vectorstore_dir)
        shutil.rmtree(settings.vectorstore_dir)
        settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    documents = load_pdfs(settings.data_dir)
    if not documents:
        return 0

    chunks = split_documents(documents)
    logger.info("Indexing %s chunks into Chroma.", len(chunks))

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(settings),
        collection_name=settings.collection_name,
        persist_directory=str(settings.vectorstore_dir),
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )
    logger.info("Vectorstore ready at %s.", settings.vectorstore_dir)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Cobb County RAG vector index.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the existing Chroma index.")
    args = parser.parse_args()
    count = build_vectorstore(rebuild=args.rebuild)
    print(f"Indexed {count} chunks.")


if __name__ == "__main__":
    main()
