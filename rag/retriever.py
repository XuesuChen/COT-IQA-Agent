"""FAISS retriever for the COT-IQA-Agent knowledge base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.build_index import (
    CHUNKS_FILENAME,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    load_json,
    load_jsonl,
    load_rag_settings,
)


class IQARetriever:
    """Load and query a persisted IQA paper index."""

    def __init__(
        self,
        *,
        vector_store_dir: str | Path | None = None,
        embedding_model: str | None = None,
        device: str = "auto",
    ) -> None:
        settings = load_rag_settings(
            vector_store_dir=(
                vector_store_dir
            ),
            embedding_model=embedding_model,
            device=device,
        )

        self.vector_store_dir = (
            settings.vector_store_dir
        )

        self.device = settings.device
        self._configured_embedding_model = (
            settings.embedding_model
        )

        self.manifest_path = (
            self.vector_store_dir
            / MANIFEST_FILENAME
        )

        self.chunks_path = (
            self.vector_store_dir
            / CHUNKS_FILENAME
        )

        self.index_path = (
            self.vector_store_dir
            / INDEX_FILENAME
        )

        self.manifest = load_json(
            self.manifest_path
        )

        self.chunks = load_jsonl(
            self.chunks_path
        )

        self._validate_artifacts()

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.embedding_model_name = str(
            self.manifest.get(
                "embedding_model"
            )
            or self._configured_embedding_model
        )

        self.embedding_dimension = int(
            self.manifest[
                "embedding_dimension"
            ]
        )

        self.normalize_embeddings = bool(
            self.manifest.get(
                "normalize_embeddings",
                True,
            )
        )

        self._model: (
            SentenceTransformer | None
        ) = None

        self._validate_index()

    def _validate_artifacts(self) -> None:
        """Validate persisted artifact presence."""

        missing = [
            str(path)
            for path in [
                self.manifest_path,
                self.chunks_path,
                self.index_path,
            ]
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "RAG index artifacts are missing: "
                + ", ".join(missing)
                + ". Run `python rag/build_index.py` first."
            )

        if not self.manifest:
            raise ValueError(
                "RAG manifest is empty or invalid."
            )

        if not self.chunks:
            raise ValueError(
                "RAG chunk store is empty."
            )

    def _validate_index(self) -> None:
        """Validate FAISS/chunk/manifest consistency."""

        manifest_chunk_count = int(
            self.manifest.get(
                "chunk_count",
                -1,
            )
        )

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index and chunk store disagree: "
                f"{self.index.ntotal} vectors versus "
                f"{len(self.chunks)} chunks."
            )

        if (
            manifest_chunk_count
            != len(self.chunks)
        ):
            raise ValueError(
                "Manifest and chunk store disagree: "
                f"{manifest_chunk_count} versus "
                f"{len(self.chunks)}."
            )

        if self.index.d != self.embedding_dimension:
            raise ValueError(
                "FAISS dimension and manifest disagree: "
                f"{self.index.d} versus "
                f"{self.embedding_dimension}."
            )

    @property
    def model(self) -> SentenceTransformer:
        """Load the embedding model lazily."""

        if self._model is None:
            self._model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
            )

        return self._model

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Encode one retrieval query."""

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query must not be empty."
            )

        vector = self.model.encode(
            [normalized_query],
            convert_to_numpy=True,
            normalize_embeddings=(
                self.normalize_embeddings
            ),
            show_progress_bar=False,
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.shape != (
            1,
            self.embedding_dimension,
        ):
            raise ValueError(
                "Query embedding has unexpected shape: "
                f"{vector.shape}; expected "
                f"(1, {self.embedding_dimension})."
            )

        if self.normalize_embeddings:
            faiss.normalize_L2(vector)

        return np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        score_threshold: float | None = None,
        deduplicate_pages: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant paper chunks."""

        if top_k < 1:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_vector = self.encode_query(
            query
        )

        candidate_count = min(
            max(
                top_k * 4
                if deduplicate_pages
                else top_k,
                top_k,
            ),
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_vector,
            candidate_count,
        )

        results: list[dict[str, Any]] = []
        seen_pages: set[
            tuple[str, int]
        ] = set()

        for score, chunk_index in zip(
            scores[0],
            indices[0],
        ):
            if chunk_index < 0:
                continue

            numeric_score = float(score)

            if (
                score_threshold is not None
                and numeric_score
                < score_threshold
            ):
                continue

            record = self.chunks[
                int(chunk_index)
            ]

            metadata = dict(
                record.get(
                    "metadata",
                    {},
                )
            )

            source_path = str(
                metadata.get(
                    "source_path",
                    ""
                )
            )

            page_number = int(
                metadata.get(
                    "page_number",
                    0,
                )
                or 0
            )

            page_key = (
                source_path,
                page_number,
            )

            if (
                deduplicate_pages
                and page_key in seen_pages
            ):
                continue

            seen_pages.add(page_key)

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": numeric_score,
                    "similarity": (
                        "cosine"
                        if self.normalize_embeddings
                        else "inner_product"
                    ),
                    "chunk_id": record.get(
                        "chunk_id"
                    ),
                    "text": record.get(
                        "text",
                        "",
                    ),
                    "metadata": metadata,
                    "citation": (
                        f"{source_path}"
                        f"#page={page_number}"
                    ),
                }
            )

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def format_context(
        results: list[dict[str, Any]],
        *,
        max_characters: int = 6000,
    ) -> str:
        """Format retrieved chunks for an Agent/LLM prompt."""

        sections: list[str] = []
        used_characters = 0

        for result in results:
            metadata = (
                result.get("metadata", {})
                or {}
            )

            section = (
                f"[Source {result.get('rank')}]\n"
                f"Title: {metadata.get('title', '')}\n"
                f"File: {metadata.get('source_path', '')}\n"
                f"Page: {metadata.get('page_number', '')}\n"
                f"Similarity: {result.get('score', 0.0):.4f}\n"
                f"Content:\n{result.get('text', '')}"
            )

            projected_length = (
                used_characters
                + len(section)
                + 2
            )

            if (
                sections
                and projected_length
                > max_characters
            ):
                break

            sections.append(section)
            used_characters = (
                projected_length
            )

        return "\n\n".join(sections)

    def unload(self) -> None:
        """Release the embedding model reference."""

        self._model = None


_RETRIEVER_CACHE: dict[
    tuple[str, str, str],
    IQARetriever,
] = {}


def get_retriever(
    *,
    vector_store_dir: str | Path | None = None,
    embedding_model: str | None = None,
    device: str = "auto",
) -> IQARetriever:
    """Get or create a cached retriever."""

    settings = load_rag_settings(
        vector_store_dir=vector_store_dir,
        embedding_model=embedding_model,
        device=device,
    )

    cache_key = (
        str(settings.vector_store_dir),
        settings.embedding_model,
        settings.device,
    )

    retriever = _RETRIEVER_CACHE.get(
        cache_key
    )

    if retriever is None:
        retriever = IQARetriever(
            vector_store_dir=(
                settings.vector_store_dir
            ),
            embedding_model=(
                settings.embedding_model
            ),
            device=settings.device,
        )

        _RETRIEVER_CACHE[
            cache_key
        ] = retriever

    return retriever


def retrieve(
    query: str,
    *,
    top_k: int = 4,
    score_threshold: float | None = None,
    deduplicate_pages: bool = False,
) -> list[dict[str, Any]]:
    """Convenience retrieval function."""

    return get_retriever().search(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        deduplicate_pages=(
            deduplicate_pages
        ),
    )


def clear_retriever_cache() -> None:
    """Unload and clear cached retrievers."""

    for retriever in (
        _RETRIEVER_CACHE.values()
    ):
        retriever.unload()

    _RETRIEVER_CACHE.clear()


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the retrieval CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Search the COT-IQA-Agent paper index."
        )
    )

    parser.add_argument(
        "query",
        help="Natural-language retrieval query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--deduplicate-pages",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--show-full-text",
        action="store_true",
    )

    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()

    retriever = IQARetriever(
        device=args.device,
    )

    results = retriever.search(
        args.query,
        top_k=args.top_k,
        score_threshold=(
            args.score_threshold
        ),
        deduplicate_pages=(
            args.deduplicate_pages
        ),
    )

    print("===== RETRIEVAL RESULTS =====")
    print("Query:", args.query)
    print("Result count:", len(results))

    for result in results:
        metadata = result["metadata"]

        print("\n" + "-" * 72)
        print("Rank:", result["rank"])
        print(
            "Score:",
            f"{result['score']:.6f}",
        )
        print(
            "Title:",
            metadata.get("title"),
        )
        print(
            "File:",
            metadata.get(
                "source_path"
            ),
        )
        print(
            "Page:",
            metadata.get(
                "page_number"
            ),
        )
        print(
            "Chunk ID:",
            result.get("chunk_id"),
        )

        text = str(
            result.get("text", "")
        )

        if not args.show_full_text:
            text = (
                text[:500]
                + (
                    "..."
                    if len(text) > 500
                    else ""
                )
            )

        print("Text:")
        print(text)

    print("\n===== JSON =====")
    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
