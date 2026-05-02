"""Build or rebuild the local Chroma vector index from PDF documents."""

from __future__ import annotations

import os
# MUST be set before importing chromadb or langchain_chroma
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import logging
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import DOCLING_COLLECTION_NAME, ORIGINAL_COLLECTION_NAME, get_embeddings, get_settings


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
            pages = load_pdf_with_original_parser(pdf_path, data_dir, parser_type="original")
            documents.extend(pages)
            logger.info("Loaded %s pages from %s.", len(pages), pdf_path.name)
        except Exception as exc:
            logger.exception("Failed to load %s: %s", pdf_path, exc)

    return documents


def load_pdf_with_original_parser(pdf_path: Path, data_dir: Path, parser_type: str) -> list[Document]:
    """Load one PDF with the original parser and annotate parser metadata."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = str(pdf_path.relative_to(data_dir.parent))
        page.metadata["file_name"] = pdf_path.name
        page.metadata["parser_type"] = parser_type
    return pages


def load_pdf_page_range_with_original_parser(
    pdf_path: Path,
    data_dir: Path,
    start_page: int,
    end_page: int,
    parser_type: str,
) -> list[Document]:
    """Load a 1-indexed inclusive PDF page range with pypdf as a fallback."""
    documents: list[Document] = []
    reader = PdfReader(str(pdf_path))
    for page_number in range(start_page, min(end_page, len(reader.pages)) + 1):
        page = reader.pages[page_number - 1]
        documents.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={
                    "source": str(pdf_path.relative_to(data_dir.parent)),
                    "file_name": pdf_path.name,
                    "page": page_number - 1,
                    "page_start": page_number,
                    "page_end": page_number,
                    "parser_type": parser_type,
                },
            )
        )
    return documents


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF, or 0 if counting fails."""
    try:
        return len(PdfReader(str(pdf_path)).pages)
    except Exception as exc:
        logger.warning("Could not count pages for %s: %s", pdf_path, exc)
        return 0


DoclingPageRange = tuple[int, int, str]


def fixed_page_ranges(
    page_count: int,
    page_chunk_size: int,
    overlap: int = 0,
    label_prefix: str = "pages",
) -> list[DoclingPageRange]:
    """Return 1-indexed inclusive page ranges, optionally overlapping adjacent windows."""
    if page_count <= 0:
        return [(1, 2**31 - 1, label_prefix)]
    chunk_size = max(page_chunk_size, 1)
    overlap = max(min(overlap, chunk_size - 1), 0)
    ranges: list[DoclingPageRange] = []
    start_page = 1
    while start_page <= page_count:
        end_page = min(start_page + chunk_size - 1, page_count)
        ranges.append((start_page, end_page, f"{label_prefix} {start_page}-{end_page}"))
        if end_page == page_count:
            break
        start_page = end_page - overlap + 1
    return ranges


def split_absolute_page_range(
    start_page: int,
    end_page: int,
    label: str,
    max_pages: int,
    overlap: int,
) -> list[DoclingPageRange]:
    """Split one absolute 1-indexed range only when it exceeds the Docling page budget."""
    if end_page < start_page:
        return []
    section_length = end_page - start_page + 1
    if section_length <= max_pages:
        return [(start_page, end_page, label)]

    section_ranges = fixed_page_ranges(
        section_length,
        max_pages,
        overlap=overlap,
        label_prefix=label,
    )
    return [
        (start_page + section_start - 1, start_page + section_end - 1, section_label)
        for section_start, section_end, section_label in section_ranges
    ]


def toc_page_ranges(pdf_path: Path, page_count: int, max_pages: int, overlap: int) -> list[DoclingPageRange]:
    """Build logical Docling page ranges from PyMuPDF bookmarks or table of contents."""
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF is not installed; falling back to fixed Docling page windows.")
        return []

    try:
        with fitz.open(str(pdf_path)) as pdf:
            toc = pdf.get_toc(simple=True)
    except Exception as exc:
        logger.warning("Could not read PyMuPDF TOC for %s: %s", pdf_path.name, exc)
        return []

    if not toc:
        return []

    bookmarks_by_level: dict[int, list[tuple[int, str]]] = {}
    for level in sorted({item[0] for item in toc}):
        bookmarks_by_level[level] = sorted(
            {
                max(min(page, page_count), 1): title.strip() or "section"
                for item_level, title, page in toc
                if item_level == level and (page_count <= 0 or page <= page_count)
            }.items()
        )

    bookmarks: list[tuple[int, str]] = []
    title_priority = ("subject ", "ch. ", "chapter ", "art. ", "article ")
    for priority in title_priority:
        for level in sorted(bookmarks_by_level):
            level_bookmarks = bookmarks_by_level[level]
            has_priority_title = any(title.lower().lstrip().startswith(priority) for _, title in level_bookmarks)
            if len(level_bookmarks) >= 3 and has_priority_title:
                bookmarks = level_bookmarks
                break
        if bookmarks:
            break

    if not bookmarks:
        for level in sorted(bookmarks_by_level):
            level_bookmarks = bookmarks_by_level[level]
            if len(level_bookmarks) >= 3:
                bookmarks = level_bookmarks
                break
            if not bookmarks and level_bookmarks:
                bookmarks = level_bookmarks

    if not bookmarks:
        return []

    ranges: list[DoclingPageRange] = []
    if bookmarks[0][0] > 1:
        ranges.extend(split_absolute_page_range(1, bookmarks[0][0] - 1, "front matter", max_pages, overlap))

    for index, (start_page, title) in enumerate(bookmarks):
        end_page = bookmarks[index + 1][0] - 1 if index + 1 < len(bookmarks) else page_count
        ranges.extend(split_absolute_page_range(start_page, end_page, title, max_pages, overlap))

    return ranges


def docling_page_ranges(pdf_path: Path, page_count: int, page_chunk_size: int, overlap: int) -> list[DoclingPageRange]:
    """Return logical page ranges for Docling conversion, preferring PDF TOC/bookmarks."""
    if page_count <= 0:
        return [(1, 2**31 - 1, "full document")]

    toc_ranges = toc_page_ranges(pdf_path, page_count, page_chunk_size, overlap)
    if toc_ranges:
        logger.info("Using %s PyMuPDF TOC/bookmark ranges for %s.", len(toc_ranges), pdf_path.name)
        return toc_ranges

    logger.info("No usable TOC/bookmarks found for %s; using overlapping fixed page windows.", pdf_path.name)
    return fixed_page_ranges(page_count, page_chunk_size, overlap=overlap)


def convert_pdf_range_with_docling(
    converter,
    pdf_path: Path,
    data_dir: Path,
    start_page: int,
    end_page: int,
    section: str,
) -> Document | None:
    """Convert one PDF page range with Docling and return a Markdown document."""
    result = converter.convert(str(pdf_path), page_range=(start_page, end_page))
    markdown = result.document.export_to_markdown()
    if not markdown.strip():
        return None
    return Document(
        page_content=markdown,
        metadata={
            "source": str(pdf_path.relative_to(data_dir.parent)),
            "file_name": pdf_path.name,
            "parser_type": "docling",
            "page_start": start_page,
            "page_end": end_page,
            "section": section,
        },
    )


def load_pdfs_with_docling(data_dir: Path) -> list[Document]:
    """Load PDFs through Docling and return layout-aware Markdown documents."""
    settings = get_settings()
    try:
        import torch
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed. Run `pip install -r requirements.txt` before building the Docling index."
        ) from exc

    documents: list[Document] = []
    pdf_files = find_pdf_files(data_dir)
    if not pdf_files:
        logger.warning("No PDF files found under %s.", data_dir)
        return documents

    requested_device = settings.docling_accelerator_device
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "DOCLING_ACCELERATOR_DEVICE=cuda was requested, but CUDA is not available to PyTorch. "
            "Docling will fall back to auto device selection."
        )
        requested_device = "auto"

    if requested_device == "cuda":
        logger.info("Docling CUDA device: %s.", torch.cuda.get_device_name(0))
    logger.info(
        "Docling accelerator device=%s, num_threads=%s, do_ocr=%s, batch_size=%s.",
        requested_device,
        settings.docling_num_threads,
        settings.docling_do_ocr,
        settings.docling_batch_size,
    )

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=requested_device,
            num_threads=settings.docling_num_threads,
        ),
        do_ocr=settings.docling_do_ocr,
        layout_batch_size=settings.docling_batch_size,
        table_batch_size=settings.docling_batch_size,
        ocr_batch_size=settings.docling_batch_size,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    for pdf_path in pdf_files:
        try:
            page_count = count_pdf_pages(pdf_path)
            page_ranges = (
                docling_page_ranges(
                    pdf_path,
                    page_count,
                    settings.docling_page_chunk_size,
                    settings.docling_page_overlap,
                )
                if page_count > settings.docling_max_pages
                else [(1, page_count or 2**31 - 1, "full document")]
            )
            if len(page_ranges) > 1:
                logger.info(
                    "Docling will process %s in %s logical/overlapping page ranges of up to %s pages.",
                    pdf_path.name,
                    len(page_ranges),
                    settings.docling_page_chunk_size,
                )

            parsed_ranges = 0
            for start_page, end_page, section in page_ranges:
                try:
                    document = convert_pdf_range_with_docling(
                        converter,
                        pdf_path,
                        data_dir,
                        start_page,
                        end_page,
                        section,
                    )
                    if document is None:
                        logger.warning(
                            "Docling returned no text for %s pages %s-%s (%s).",
                            pdf_path.name,
                            start_page,
                            end_page,
                            section,
                        )
                        continue
                    documents.append(document)
                    parsed_ranges += 1
                    logger.info(
                        "Docling parsed %s pages %s-%s (%s).",
                        pdf_path.name,
                        start_page,
                        end_page,
                        section,
                    )
                except Exception as range_exc:
                    logger.exception(
                        "Docling failed for %s pages %s-%s (%s); using original parser fallback: %s",
                        pdf_path.name,
                        start_page,
                        end_page,
                        section,
                        range_exc,
                    )
                    documents.extend(
                        load_pdf_page_range_with_original_parser(
                            pdf_path,
                            data_dir,
                            start_page=start_page,
                            end_page=end_page,
                            parser_type="docling_page_range_fallback_original",
                        )
                    )

            logger.info("Docling parsed %s page ranges from %s.", parsed_ranges, pdf_path.name)
        except Exception as exc:
            logger.exception("Docling failed to parse %s: %s", pdf_path, exc)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into retrieval chunks."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def delete_collection(collection_name: str) -> None:
    """Delete one Chroma collection if it exists."""
    settings = get_settings()
    if not settings.vectorstore_dir.exists():
        return
    
    # settings argument is optional in modern Chroma but kept for explicit control
    client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing Chroma collection %s.", collection_name)
    except Exception:
        logger.info("No existing Chroma collection named %s to delete.", collection_name)


def index_documents(documents: list[Document], collection_name: str, rebuild: bool = False) -> int:
    """Embed documents into the requested Chroma collection."""
    settings = get_settings()
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    if rebuild:
        delete_collection(collection_name)
    if not documents:
        return 0

    chunks = split_documents(documents)
    logger.info("Indexing %s chunks into Chroma collection %s.", len(chunks), collection_name)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(settings),
        collection_name=collection_name,
        persist_directory=str(settings.vectorstore_dir),
    )
    logger.info("Vectorstore collection %s ready at %s.", collection_name, settings.vectorstore_dir)
    return len(chunks)


def build_original_vectorstore(rebuild: bool = False) -> int:
    """Build the original PyPDFLoader-based Chroma collection."""
    settings = get_settings()
    documents = load_pdfs(settings.data_dir)
    return index_documents(documents, ORIGINAL_COLLECTION_NAME, rebuild=rebuild)


def build_docling_vectorstore(rebuild: bool = False) -> int:
    """Build the Docling-enhanced Chroma collection."""
    settings = get_settings()
    documents = load_pdfs_with_docling(settings.data_dir)
    return index_documents(documents, DOCLING_COLLECTION_NAME, rebuild=rebuild)


def build_vectorstore(rebuild: bool = False, pipeline: str = "original") -> int:
    """Build one or both Chroma collections and return the indexed chunk count."""
    settings = get_settings()
    if rebuild and pipeline == "both" and settings.vectorstore_dir.exists():
        logger.info("Rebuilding all vectorstore collections at %s.", settings.vectorstore_dir)
        shutil.rmtree(settings.vectorstore_dir)
        settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    if pipeline == "original":
        return build_original_vectorstore(rebuild=rebuild)
    if pipeline == "docling":
        return build_docling_vectorstore(rebuild=rebuild)
    if pipeline == "both":
        original_count = build_original_vectorstore(rebuild=False)
        docling_count = build_docling_vectorstore(rebuild=False)
        return original_count + docling_count
    raise ValueError(f"Unsupported ingestion pipeline: {pipeline}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Cobb County RAG vector index.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the existing Chroma index.")
    parser.add_argument(
        "--pipeline",
        choices=["original", "docling", "both"],
        default="original",
        help="Which ingestion pipeline to run. Default preserves the original behavior.",
    )
    parser.add_argument(
        "--docling-device",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="Override DOCLING_ACCELERATOR_DEVICE for this ingestion run.",
    )
    args = parser.parse_args()
    if args.docling_device:
        os.environ["DOCLING_ACCELERATOR_DEVICE"] = args.docling_device
    
    count = build_vectorstore(rebuild=args.rebuild, pipeline=args.pipeline)
    print(f"Indexed {count} chunks with pipeline={args.pipeline}.")


if __name__ == "__main__":
    main()