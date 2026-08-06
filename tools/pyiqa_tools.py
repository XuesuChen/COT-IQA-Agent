"""PyIQA metric tools for image-quality evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pyiqa
import torch

from tools.image_tools import validate_image


DEFAULT_METRICS = ("niqe", "brisque")

# Cache loaded metric objects to avoid repeatedly loading them.
_METRIC_CACHE: dict[tuple[str, str], Any] = {}


def resolve_device(device: str = "auto") -> str:
    """Resolve auto/cpu/cuda into an available PyTorch device."""

    normalized = device.strip().lower()

    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    if normalized not in {"cpu", "cuda"}:
        raise ValueError(
            "device must be one of: auto, cpu, cuda."
        )

    return normalized


def list_available_metrics() -> list[str]:
    """Return metric names supported by the installed PyIQA version."""
    return sorted(pyiqa.list_models())


def _load_metric(metric_name: str, device: str) -> Any:
    """Load and cache one PyIQA metric."""

    normalized_name = metric_name.strip().lower()

    if not normalized_name:
        raise ValueError("metric_name must not be empty.")

    available_metrics = set(pyiqa.list_models())

    if normalized_name not in available_metrics:
        raise ValueError(
            f"Unknown PyIQA metric: {normalized_name}. "
            f"Available metric count: {len(available_metrics)}."
        )

    cache_key = (normalized_name, device)

    if cache_key not in _METRIC_CACHE:
        _METRIC_CACHE[cache_key] = pyiqa.create_metric(
            normalized_name,
            device=device,
        )

    return _METRIC_CACHE[cache_key]


def _score_to_python(score: Any) -> float | list[float]:
    """Convert a PyTorch/PyIQA result into JSON-serializable values."""

    if isinstance(score, torch.Tensor):
        values = score.detach().cpu().reshape(-1).tolist()

        if len(values) == 1:
            return float(values[0])

        return [float(value) for value in values]

    if isinstance(score, (int, float)):
        return float(score)

    raise TypeError(
        f"Unsupported score type: {type(score).__name__}"
    )


def evaluate_image(
    image_path: str | Path,
    metrics: Sequence[str] | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate one image using one or more no-reference IQA metrics."""

    validation = validate_image(image_path)

    if not validation["valid"]:
        return {
            "path": validation["path"],
            "valid": False,
            "device": None,
            "results": {},
            "errors": [
                validation.get("error") or "Invalid image."
            ],
        }

    resolved_device = resolve_device(device)
    selected_metrics = tuple(metrics or DEFAULT_METRICS)

    results: dict[str, Any] = {}
    errors: list[str] = []

    for metric_name in selected_metrics:
        normalized_name = metric_name.strip().lower()

        try:
            metric = _load_metric(
                metric_name=normalized_name,
                device=resolved_device,
            )

            with torch.inference_mode():
                score = metric(validation["path"])

            results[normalized_name] = {
                "score": _score_to_python(score),
                "lower_better": bool(metric.lower_better),
            }

        except Exception as exc:
            errors.append(
                f"{normalized_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "path": validation["path"],
        "valid": True,
        "device": resolved_device,
        "results": results,
        "errors": errors,
    }


def evaluate_images(
    image_paths: Sequence[str | Path],
    metrics: Sequence[str] | None = None,
    device: str = "auto",
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple images without stopping after one failure."""

    return {
        str(image_path): evaluate_image(
            image_path=image_path,
            metrics=metrics,
            device=device,
        )
        for image_path in image_paths
    }


def compare_images(
    image_paths: Sequence[str | Path],
    metrics: Sequence[str] | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Rank images independently for each selected IQA metric."""

    evaluations = evaluate_images(
        image_paths=image_paths,
        metrics=metrics,
        device=device,
    )

    selected_metrics = tuple(metrics or DEFAULT_METRICS)
    rankings: dict[str, list[dict[str, Any]]] = {}

    for metric_name in selected_metrics:
        normalized_name = metric_name.strip().lower()
        metric_entries: list[dict[str, Any]] = []
        lower_better: bool | None = None

        for path, evaluation in evaluations.items():
            metric_result = evaluation["results"].get(normalized_name)

            if metric_result is None:
                continue

            score = metric_result["score"]

            if not isinstance(score, float):
                continue

            lower_better = metric_result["lower_better"]

            metric_entries.append(
                {
                    "path": path,
                    "score": score,
                }
            )

        if lower_better is not None:
            metric_entries.sort(
                key=lambda item: item["score"],
                reverse=not lower_better,
            )

        rankings[normalized_name] = metric_entries

    return {
        "evaluations": evaluations,
        "rankings": rankings,
    }


def clear_metric_cache() -> None:
    """Release cached metric objects and optional CUDA cache."""

    _METRIC_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
