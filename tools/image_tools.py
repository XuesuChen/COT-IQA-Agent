"""Image validation, metadata extraction, and basic statistics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def _normalize_path(image_path: str | Path) -> Path:
    """Normalize and validate an image-path argument."""
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path must be a string or pathlib.Path.")

    path = Path(image_path).expanduser()

    if not str(path).strip():
        raise ValueError("image_path must not be empty.")

    return path.resolve()


def validate_image(image_path: str | Path) -> dict[str, Any]:
    """Check whether an image exists, is supported, and can be decoded."""

    try:
        path = _normalize_path(image_path)
    except (TypeError, ValueError) as exc:
        return {
            "valid": False,
            "path": str(image_path),
            "error": str(exc),
        }

    result: dict[str, Any] = {
        "valid": False,
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "extension": path.suffix.lower(),
        "extension_supported": path.suffix.lower() in SUPPORTED_EXTENSIONS,
        "format": None,
        "width": None,
        "height": None,
        "mode": None,
        "frame_count": None,
        "error": None,
    }

    if not path.exists():
        result["error"] = "Image file does not exist."
        return result

    if not path.is_file():
        result["error"] = "The supplied path is not a file."
        return result

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["error"] = (
            f"Unsupported image extension: {path.suffix.lower() or '<none>'}"
        )
        return result

    try:
        # verify() checks integrity but invalidates the current image object.
        with Image.open(path) as image:
            image_format = image.format
            image.verify()

        # Reopen to read metadata after verify().
        with Image.open(path) as image:
            width, height = image.size

            if width <= 0 or height <= 0:
                result["error"] = "Image dimensions must be positive."
                return result

            result.update(
                {
                    "valid": True,
                    "format": image_format,
                    "width": width,
                    "height": height,
                    "mode": image.mode,
                    "frame_count": getattr(image, "n_frames", 1),
                }
            )

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        result["error"] = f"Image decoding failed: {exc}"

    return result


def load_image(
    image_path: str | Path,
    mode: str = "RGB",
) -> Image.Image:
    """Load an image and return an independent PIL image object."""

    validation = validate_image(image_path)

    if not validation["valid"]:
        raise ValueError(validation.get("error") or "Invalid image.")

    path = Path(validation["path"])

    with Image.open(path) as image:
        loaded = image.convert(mode).copy()

    return loaded


def extract_image_metadata(image_path: str | Path) -> dict[str, Any]:
    """Extract file metadata and selected EXIF information."""

    validation = validate_image(image_path)

    if not validation["valid"]:
        raise ValueError(validation.get("error") or "Invalid image.")

    path = Path(validation["path"])

    with Image.open(path) as image:
        width, height = image.size
        exif_data = image.getexif()

        selected_exif: dict[str, Any] = {}

        selected_names = {
            "Make",
            "Model",
            "Software",
            "Orientation",
            "DateTime",
            "DateTimeOriginal",
        }

        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))

            if tag_name in selected_names:
                selected_exif[tag_name] = str(value)

        return {
            "path": str(path),
            "filename": path.name,
            "format": image.format,
            "mode": image.mode,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 6),
            "megapixels": round((width * height) / 1_000_000, 4),
            "file_size_bytes": path.stat().st_size,
            "file_size_kb": round(path.stat().st_size / 1024, 3),
            "frame_count": getattr(image, "n_frames", 1),
            "has_exif": bool(exif_data),
            "exif": selected_exif,
        }


def _calculate_entropy(gray_uint8: np.ndarray) -> float:
    """Calculate Shannon entropy of an 8-bit grayscale image."""

    histogram = np.bincount(
        gray_uint8.ravel(),
        minlength=256,
    ).astype(np.float64)

    total = histogram.sum()

    if total == 0:
        return 0.0

    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0]

    return float(
        -np.sum(probabilities * np.log2(probabilities))
    )


def analyze_image_statistics(image_path: str | Path) -> dict[str, Any]:
    """Compute lightweight descriptive image statistics.

    These statistics are not perceptual IQA scores. They are auxiliary
    signals for later Agent reasoning and result verification.
    """

    image = load_image(image_path, mode="RGB")
    rgb = np.asarray(image, dtype=np.float32)

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

    channel_means = rgb.mean(axis=(0, 1))
    channel_stds = rgb.std(axis=(0, 1))

    p01 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))

    horizontal_gradient = np.abs(np.diff(gray, axis=1))
    vertical_gradient = np.abs(np.diff(gray, axis=0))

    gradient_values = []

    if horizontal_gradient.size:
        gradient_values.append(float(horizontal_gradient.mean()))

    if vertical_gradient.size:
        gradient_values.append(float(vertical_gradient.mean()))

    edge_strength = (
        float(np.mean(gradient_values))
        if gradient_values
        else 0.0
    )

    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    saturation = hsv[:, :, 1] / 255.0

    dark_clip_ratio = float(np.mean(gray <= 5.0))
    bright_clip_ratio = float(np.mean(gray >= 250.0))

    values = {
        "brightness_mean": float(gray.mean()),
        "brightness_normalized": float(gray.mean() / 255.0),
        "contrast_std": float(gray.std()),
        "gray_min": float(gray.min()),
        "gray_max": float(gray.max()),
        "robust_dynamic_range": p99 - p01,
        "entropy": _calculate_entropy(gray_uint8),
        "saturation_mean": float(saturation.mean()),
        "edge_strength": edge_strength,
        "dark_clip_ratio": dark_clip_ratio,
        "bright_clip_ratio": bright_clip_ratio,
        "red_mean": float(channel_means[0]),
        "green_mean": float(channel_means[1]),
        "blue_mean": float(channel_means[2]),
        "red_std": float(channel_stds[0]),
        "green_std": float(channel_stds[1]),
        "blue_std": float(channel_stds[2]),
    }

    return {
        key: round(value, 6)
        if isinstance(value, float) and math.isfinite(value)
        else value
        for key, value in values.items()
    }


def inspect_image(image_path: str | Path) -> dict[str, Any]:
    """Run validation, metadata extraction, and statistical inspection."""

    validation = validate_image(image_path)

    if not validation["valid"]:
        return {
            "validation": validation,
            "metadata": {},
            "statistics": {},
        }

    return {
        "validation": validation,
        "metadata": extract_image_metadata(image_path),
        "statistics": analyze_image_statistics(image_path),
    }


def inspect_images(
    image_paths: list[str | Path],
) -> dict[str, dict[str, Any]]:
    """Inspect multiple images without stopping on one invalid input."""

    results: dict[str, dict[str, Any]] = {}

    for image_path in image_paths:
        results[str(image_path)] = inspect_image(image_path)

    return results
