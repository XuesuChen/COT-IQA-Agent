"""Build an incremental FAISS index from IQA research papers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import faiss
import fitz
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


SCHEMA_VERSION = 1
INDEX_FILENAME = "index.faiss"
EMBEDDINGS_FILENAME = "embeddings.npy"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class RAGSettings:
    """Resolved RAG indexing configuration."""

    project_root: Path
    paper_dir: Path
    vector_store_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    normalize_embeddings: bool
    batch_size: int
    device: str


def _resolve_env_placeholders(value: Any) -> Any:
    """Resolve ${VARIABLE} placeholders recursively."""

    if isinstance(value, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def replace(match: re.Match[str]) -> str:
            variable_name = match.group(1)
            return os.getenv(variable_name, match.group(0))

        return pattern.sub(replace, value)

    if isinstance(value, list):
        return [
            _resolve_env_placeholders(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _resolve_env_placeholders(item)
            for key, item in value.items()
        }

    return value


def _resolve_path(
    project_root: Path,
    value: str | Path,
) -> Path:
    """Resolve a project-relative or absolute path."""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def _resolve_device(device: str) -> str:
    """Resolve auto device selection."""

    normalized = str(device).strip().lower()

    if normalized in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"

    return normalized


def load_rag_settings(
    *,
    project_root: str | Path | None = None,
    paper_dir: str | Path | None = None,
    vector_store_dir: str | Path | None = None,
    embedding_model: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int = 16,
    device: str = "auto",
) -> RAGSettings:
    """Load RAG settings from .env and configs/config.yaml."""

    resolved_root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )

    load_dotenv(
        resolved_root / ".env",
        override=False,
    )

    config_path = (
        resolved_root
        / "configs"
        / "config.yaml"
    )

    config: dict[str, Any] = {}

    if config_path.is_file():
        loaded = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

        if isinstance(loaded, dict):
            config = _resolve_env_placeholders(loaded)

    paths_config = config.get("paths", {}) or {}
    rag_config = config.get("rag", {}) or {}

    resolved_paper_dir = _resolve_path(
        resolved_root,
        paper_dir
        or paths_config.get("paper_dir")
        or os.getenv("PAPER_DIR")
        or "assets/papers",
    )

    resolved_vector_dir = _resolve_path(
        resolved_root,
        vector_store_dir
        or paths_config.get("vector_store_dir")
        or os.getenv("VECTOR_STORE_DIR")
        or "rag/vector_store",
    )

    resolved_embedding_model = str(
        embedding_model
        or rag_config.get("embedding_model")
        or os.getenv("EMBEDDING_MODEL")
        or "BAAI/bge-m3"
    ).strip()

    if (
        not resolved_embedding_model
        or resolved_embedding_model.startswith("${")
    ):
        resolved_embedding_model = "BAAI/bge-m3"

    resolved_chunk_size = int(
        chunk_size
        if chunk_size is not None
        else rag_config.get("chunk_size", 800)
    )

    resolved_chunk_overlap = int(
        chunk_overlap
        if chunk_overlap is not None
        else rag_config.get("chunk_overlap", 120)
    )

    normalize_embeddings = bool(
        rag_config.get(
            "normalize_embeddings",
            True,
        )
    )

    if resolved_chunk_size < 100:
        raise ValueError(
            "chunk_size must be at least 100 characters."
        )

    if not (
        0
        <= resolved_chunk_overlap
        < resolved_chunk_size
    ):
        raise ValueError(
            "chunk_overlap must satisfy "
            "0 <= chunk_overlap < chunk_size."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    return RAGSettings(
        project_root=resolved_root,
        paper_dir=resolved_paper_dir,
        vector_store_dir=resolved_vector_dir,
        embedding_model=resolved_embedding_model,
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        normalize_embeddings=normalize_embeddings,
        batch_size=int(batch_size),
        device=_resolve_device(device),
    )


def sha256_file(
    file_path: str | Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Calculate a file SHA-256 digest."""

    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file_handle:
        while True:
            block = file_handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Calculate a UTF-8 text SHA-256 digest."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def clean_extracted_text(text: str) -> str:
    """Normalize PDF text while preserving paragraphs."""

    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", "")
        .replace("\u00ad", "")
    )

    # Join words split by line-end hyphenation.
    normalized = re.sub(
        r"(?<=\w)-\n(?=\w)",
        "",
        normalized,
    )

    paragraphs = re.split(
        r"\n\s*\n+",
        normalized,
    )

    cleaned_paragraphs: list[str] = []

    for paragraph in paragraphs:
        paragraph = re.sub(
            r"\s*\n\s*",
            " ",
            paragraph,
        )

        paragraph = re.sub(
            r"[ \t]+",
            " ",
            paragraph,
        ).strip()

        if paragraph:
            cleaned_paragraphs.append(paragraph)

    return "\n\n".join(cleaned_paragraphs)


def _extract_with_pypdf(
    pdf_path: Path,
) -> list[str]:
    """Extract all pages with pypdf as fallback."""

    try:
        reader = PdfReader(str(pdf_path))

        return [
            clean_extracted_text(
                page.extract_text() or ""
            )
            for page in reader.pages
        ]

    except Exception:
        return []


def extract_pdf_pages(
    pdf_path: str | Path,
) -> dict[str, Any]:
    """Extract page text using PyMuPDF with pypdf fallback."""

    resolved_path = Path(pdf_path).expanduser().resolve()

    fallback_pages: list[str] | None = None
    pages: list[dict[str, Any]] = []

    with fitz.open(resolved_path) as document:
        metadata = document.metadata or {}

        title = (
            str(metadata.get("title") or "").strip()
            or resolved_path.stem
        )

        for page_index in range(document.page_count):
            parser_name = "pymupdf"
            extraction_error: str | None = None

            try:
                raw_text = document.load_page(
                    page_index
                ).get_text("text")

                text = clean_extracted_text(
                    raw_text or ""
                )

            except Exception as exc:
                text = ""
                extraction_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            if len(text) < 20:
                if fallback_pages is None:
                    fallback_pages = (
                        _extract_with_pypdf(
                            resolved_path
                        )
                    )

                fallback_text = (
                    fallback_pages[page_index]
                    if page_index
                    < len(fallback_pages)
                    else ""
                )

                if len(fallback_text) > len(text):
                    text = fallback_text
                    parser_name = "pypdf"

            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text,
                    "parser": parser_name,
                    "extraction_error": extraction_error,
                }
            )

    return {
        "path": str(resolved_path),
        "title": title,
        "page_count": len(pages),
        "pages": pages,
    }


def _find_chunk_boundary(
    text: str,
    start: int,
    desired_end: int,
    minimum_end: int,
) -> int:
    """Find a readable chunk boundary near desired_end."""

    if desired_end >= len(text):
        return len(text)

    separators = [
        "\n\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
    ]

    search_region = text[
        minimum_end:desired_end
    ]

    for separator in separators:
        position = search_region.rfind(
            separator
        )

        if position >= 0:
            return (
                minimum_end
                + position
                + len(separator)
            )

    return desired_end


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Split text deterministically with character overlap."""

    if not text.strip():
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        desired_end = min(
            start + chunk_size,
            text_length,
        )

        minimum_end = min(
            start + max(
                int(chunk_size * 0.6),
                1,
            ),
            desired_end,
        )

        end = _find_chunk_boundary(
            text,
            start,
            desired_end,
            minimum_end,
        )

        if end <= start:
            end = desired_end

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "char_start": start,
                    "char_end": end,
                }
            )

        if end >= text_length:
            break

        next_start = max(
            end - chunk_overlap,
            start + 1,
        )

        start = next_start

    return chunks


def build_document_chunks(
    pdf_path: str | Path,
    *,
    paper_dir: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    file_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract and chunk one PDF with page metadata."""

    resolved_pdf = Path(
        pdf_path
    ).expanduser().resolve()

    resolved_paper_dir = Path(
        paper_dir
    ).expanduser().resolve()

    relative_path = resolved_pdf.relative_to(
        resolved_paper_dir
    ).as_posix()

    current_file_hash = (
        file_sha256
        or sha256_file(resolved_pdf)
    )

    extracted = extract_pdf_pages(
        resolved_pdf
    )

    records: list[dict[str, Any]] = []
    global_chunk_index = 0
    nonempty_page_count = 0

    for page in extracted["pages"]:
        page_text = str(
            page.get("text") or ""
        )

        if not page_text:
            continue

        nonempty_page_count += 1

        page_chunks = split_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for page_chunk_index, chunk in enumerate(
            page_chunks,
            start=1,
        ):
            chunk_text = chunk["text"]

            chunk_identity = "|".join(
                [
                    relative_path,
                    current_file_hash,
                    str(page["page_number"]),
                    str(chunk["char_start"]),
                    str(chunk["char_end"]),
                    sha256_text(chunk_text),
                ]
            )

            chunk_id = hashlib.sha256(
                chunk_identity.encode("utf-8")
            ).hexdigest()[:24]

            records.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source_path": relative_path,
                        "file_name": resolved_pdf.name,
                        "title": extracted["title"],
                        "page_number": page[
                            "page_number"
                        ],
                        "page_chunk_index": (
                            page_chunk_index
                        ),
                        "document_chunk_index": (
                            global_chunk_index
                        ),
                        "char_start": chunk[
                            "char_start"
                        ],
                        "char_end": chunk[
                            "char_end"
                        ],
                        "parser": page["parser"],
                        "file_sha256": (
                            current_file_hash
                        ),
                        "text_sha256": (
                            sha256_text(chunk_text)
                        ),
                    },
                }
            )

            global_chunk_index += 1

    file_stat = resolved_pdf.stat()

    file_manifest = {
        "source_path": relative_path,
        "file_name": resolved_pdf.name,
        "title": extracted["title"],
        "file_sha256": current_file_hash,
        "size_bytes": file_stat.st_size,
        "modified_time_ns": (
            file_stat.st_mtime_ns
        ),
        "page_count": extracted["page_count"],
        "nonempty_page_count": (
            nonempty_page_count
        ),
        "chunk_count": len(records),
    }

    return records, file_manifest


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load JSONL records."""

    resolved_path = Path(path)

    if not resolved_path.is_file():
        return []

    records: list[dict[str, Any]] = []

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        for line_number, line in enumerate(
            file_handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at "
                    f"{resolved_path}:{line_number}: "
                    f"{exc}"
                ) from exc

            records.append(payload)

    return records


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load one JSON object."""

    resolved_path = Path(path)

    if not resolved_path.is_file():
        return {}

    payload = json.loads(
        resolved_path.read_text(
            encoding="utf-8"
        )
    )

    return payload if isinstance(payload, dict) else {}


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """Atomically write UTF-8 text."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        content,
        encoding="utf-8",
    )

    temporary_path.replace(path)


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically write formatted JSON."""

    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def _atomic_write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Atomically write JSONL records."""

    content = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    )

    _atomic_write_text(
        path,
        content,
    )


def _atomic_write_numpy(
    path: Path,
    array: np.ndarray,
) -> None:
    """Atomically write a NumPy array."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open("wb") as file_handle:
        np.save(
            file_handle,
            array,
            allow_pickle=False,
        )

    temporary_path.replace(path)


def _atomic_write_faiss(
    path: Path,
    index: faiss.Index,
) -> None:
    """Atomically write a FAISS index."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    faiss.write_index(
        index,
        str(temporary_path),
    )

    temporary_path.replace(path)


def _settings_fingerprint(
    settings: RAGSettings,
) -> str:
    """Build a fingerprint for embedding/chunk settings."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": (
            settings.embedding_model
        ),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": (
            settings.chunk_overlap
        ),
        "normalize_embeddings": (
            settings.normalize_embeddings
        ),
        "embedding_template": (
            "title + page_number + chunk_text"
        ),
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return sha256_text(serialized)


def _embedding_text(
    record: dict[str, Any],
) -> str:
    """Build the text sent to the embedding model."""

    metadata = record["metadata"]

    return (
        f"Title: {metadata.get('title', '')}\n"
        f"Page: {metadata.get('page_number', '')}\n"
        f"{record['text']}"
    )


def _load_old_index_data(
    vector_store_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    np.ndarray | None,
]:
    """Load previous manifest, chunks and embeddings."""

    manifest = load_json(
        vector_store_dir / MANIFEST_FILENAME
    )

    chunks = load_jsonl(
        vector_store_dir / CHUNKS_FILENAME
    )

    embeddings_path = (
        vector_store_dir
        / EMBEDDINGS_FILENAME
    )

    embeddings: np.ndarray | None = None

    if embeddings_path.is_file():
        embeddings = np.load(
            embeddings_path,
            allow_pickle=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

    if (
        embeddings is not None
        and len(chunks) != embeddings.shape[0]
    ):
        return {}, [], None

    return manifest, chunks, embeddings


def _group_old_records(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> dict[str, tuple[list[dict[str, Any]], np.ndarray]]:
    """Group old records and vectors by source path."""

    indices_by_source: dict[str, list[int]] = {}

    for index, record in enumerate(records):
        metadata = record.get(
            "metadata",
            {},
        )

        source_path = metadata.get(
            "source_path"
        )

        if not isinstance(source_path, str):
            continue

        indices_by_source.setdefault(
            source_path,
            [],
        ).append(index)

    grouped: dict[
        str,
        tuple[list[dict[str, Any]], np.ndarray],
    ] = {}

    for source_path, indices in (
        indices_by_source.items()
    ):
        grouped[source_path] = (
            [records[index] for index in indices],
            embeddings[
                np.asarray(indices, dtype=np.int64)
            ],
        )

    return grouped


def _encode_records(
    records: list[dict[str, Any]],
    *,
    model: SentenceTransformer,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    """Embed chunk records."""

    if not records:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    texts = [
        _embedding_text(record)
        for record in records
    ]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=(
            normalize_embeddings
        ),
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError(
            "Embedding model returned a non-matrix result."
        )

    if normalize_embeddings:
        faiss.normalize_L2(vectors)

    return np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )


def discover_pdf_files(
    paper_dir: str | Path,
) -> list[Path]:
    """Find PDF files recursively and deterministically."""

    resolved_dir = Path(
        paper_dir
    ).expanduser().resolve()

    return sorted(
        (
            path.resolve()
            for path in resolved_dir.rglob("*")
            if path.is_file()
            and path.suffix.casefold() == ".pdf"
            and not any(
                part.startswith(".")
                for part in path.relative_to(
                    resolved_dir
                ).parts
            )
        ),
        key=lambda path: (
            path.relative_to(
                resolved_dir
            ).as_posix().casefold()
        ),
    )


def build_index(
    settings: RAGSettings,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build or incrementally update the RAG index."""

    settings.paper_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.vector_store_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = discover_pdf_files(
        settings.paper_dir
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found in "
            f"{settings.paper_dir}."
        )

    settings_fingerprint = (
        _settings_fingerprint(settings)
    )

    old_manifest, old_records, old_embeddings = (
        _load_old_index_data(
            settings.vector_store_dir
        )
    )

    reuse_enabled = (
        not force
        and bool(old_manifest)
        and old_embeddings is not None
        and old_manifest.get(
            "settings_fingerprint"
        )
        == settings_fingerprint
    )

    old_files = {
        item["source_path"]: item
        for item in (
            old_manifest.get("files", [])
            if reuse_enabled
            else []
        )
        if isinstance(item, dict)
        and isinstance(
            item.get("source_path"),
            str,
        )
    }

    grouped_old_records = (
        _group_old_records(
            old_records,
            old_embeddings,
        )
        if reuse_enabled
        and old_embeddings is not None
        else {}
    )

    all_records: list[dict[str, Any]] = []
    vector_blocks: list[np.ndarray] = []
    file_manifests: list[dict[str, Any]] = []

    reused_files: list[str] = []
    indexed_files: list[str] = []

    model: SentenceTransformer | None = None
    embedding_dimension: int | None = None

    for pdf_path in pdf_files:
        relative_path = pdf_path.relative_to(
            settings.paper_dir
        ).as_posix()

        file_hash = sha256_file(pdf_path)

        old_file = old_files.get(
            relative_path
        )

        old_group = grouped_old_records.get(
            relative_path
        )

        can_reuse = (
            old_file is not None
            and old_group is not None
            and old_file.get(
                "file_sha256"
            )
            == file_hash
        )

        if can_reuse:
            file_records, file_vectors = (
                old_group
            )

            all_records.extend(
                file_records
            )

            vector_blocks.append(
                np.asarray(
                    file_vectors,
                    dtype=np.float32,
                )
            )

            current_stat = pdf_path.stat()

            reused_manifest = {
                **old_file,
                "size_bytes": (
                    current_stat.st_size
                ),
                "modified_time_ns": (
                    current_stat.st_mtime_ns
                ),
            }

            file_manifests.append(
                reused_manifest
            )

            reused_files.append(
                relative_path
            )

            if file_vectors.ndim == 2:
                embedding_dimension = (
                    file_vectors.shape[1]
                )

            continue

        file_records, file_manifest = (
            build_document_chunks(
                pdf_path,
                paper_dir=settings.paper_dir,
                chunk_size=settings.chunk_size,
                chunk_overlap=(
                    settings.chunk_overlap
                ),
                file_sha256=file_hash,
            )
        )

        if not file_records:
            file_manifest[
                "indexing_warning"
            ] = (
                "No extractable text chunks "
                "were produced."
            )

            file_manifests.append(
                file_manifest
            )

            indexed_files.append(
                relative_path
            )

            continue

        all_records.extend(
            file_records
        )

        file_manifests.append(
            file_manifest
        )

        indexed_files.append(
            relative_path
        )

        if dry_run:
            continue

        if model is None:
            print(
                "Loading embedding model:",
                settings.embedding_model,
            )

            print(
                "Embedding device:",
                settings.device,
            )

            model = SentenceTransformer(
                settings.embedding_model,
                device=settings.device,
            )

        file_vectors = _encode_records(
            file_records,
            model=model,
            batch_size=settings.batch_size,
            normalize_embeddings=(
                settings.normalize_embeddings
            ),
        )

        embedding_dimension = (
            file_vectors.shape[1]
        )

        vector_blocks.append(
            file_vectors
        )

    if not all_records:
        raise RuntimeError(
            "PDF files were found, but no text chunks "
            "could be extracted."
        )

    summary = {
        "pdf_count": len(pdf_files),
        "chunk_count": len(all_records),
        "reused_file_count": len(reused_files),
        "indexed_file_count": len(indexed_files),
        "reused_files": reused_files,
        "indexed_files": indexed_files,
    }

    if dry_run:
        return {
            "status": "dry_run",
            **summary,
            "settings": {
                **asdict(settings),
                "project_root": str(
                    settings.project_root
                ),
                "paper_dir": str(
                    settings.paper_dir
                ),
                "vector_store_dir": str(
                    settings.vector_store_dir
                ),
            },
        }

    if not vector_blocks:
        raise RuntimeError(
            "No embedding vectors were generated or reused."
        )

    embeddings = np.concatenate(
        vector_blocks,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if embeddings.shape[0] != len(all_records):
        raise RuntimeError(
            "Chunk and embedding counts are inconsistent: "
            f"{len(all_records)} chunks versus "
            f"{embeddings.shape[0]} vectors."
        )

    if settings.normalize_embeddings:
        faiss.normalize_L2(embeddings)

    embedding_dimension = int(
        embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    index.add(
        np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )
    )

    built_at = datetime.now(
        timezone.utc
    ).isoformat()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": built_at,
        "index_type": "IndexFlatIP",
        "similarity": (
            "cosine"
            if settings.normalize_embeddings
            else "inner_product"
        ),
        "embedding_model": (
            settings.embedding_model
        ),
        "embedding_dimension": (
            embedding_dimension
        ),
        "normalize_embeddings": (
            settings.normalize_embeddings
        ),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": (
            settings.chunk_overlap
        ),
        "chunk_count": len(all_records),
        "document_count": len(pdf_files),
        "settings_fingerprint": (
            settings_fingerprint
        ),
        "files": file_manifests,
        "build_summary": summary,
        "artifacts": {
            "index": INDEX_FILENAME,
            "embeddings": (
                EMBEDDINGS_FILENAME
            ),
            "chunks": CHUNKS_FILENAME,
            "manifest": MANIFEST_FILENAME,
        },
    }

    _atomic_write_faiss(
        settings.vector_store_dir
        / INDEX_FILENAME,
        index,
    )

    _atomic_write_numpy(
        settings.vector_store_dir
        / EMBEDDINGS_FILENAME,
        embeddings,
    )

    _atomic_write_jsonl(
        settings.vector_store_dir
        / CHUNKS_FILENAME,
        all_records,
    )

    _atomic_write_json(
        settings.vector_store_dir
        / MANIFEST_FILENAME,
        manifest,
    )

    return {
        "status": "built",
        **summary,
        "embedding_dimension": (
            embedding_dimension
        ),
        "index_size": index.ntotal,
        "vector_store_dir": str(
            settings.vector_store_dir
        ),
        "manifest_path": str(
            settings.vector_store_dir
            / MANIFEST_FILENAME
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build an incremental FAISS index "
            "from PDF research papers."
        )
    )

    parser.add_argument(
        "--paper-dir",
        default=None,
    )

    parser.add_argument(
        "--vector-store-dir",
        default=None,
    )

    parser.add_argument(
        "--embedding-model",
        default=None,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Disable incremental reuse and rebuild all files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk PDFs without embedding or writing.",
    )

    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()

    settings = load_rag_settings(
        paper_dir=args.paper_dir,
        vector_store_dir=(
            args.vector_store_dir
        ),
        embedding_model=(
            args.embedding_model
        ),
        chunk_size=args.chunk_size,
        chunk_overlap=(
            args.chunk_overlap
        ),
        batch_size=args.batch_size,
        device=args.device,
    )

    print("===== RAG INDEX SETTINGS =====")
    print("Paper directory:", settings.paper_dir)
    print(
        "Vector directory:",
        settings.vector_store_dir,
    )
    print(
        "Embedding model:",
        settings.embedding_model,
    )
    print("Device:", settings.device)
    print("Chunk size:", settings.chunk_size)
    print(
        "Chunk overlap:",
        settings.chunk_overlap,
    )
    print(
        "Normalize embeddings:",
        settings.normalize_embeddings,
    )

    result = build_index(
        settings,
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\n===== BUILD RESULT =====")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
