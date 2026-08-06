"""LangGraph workflow nodes for COT-IQA-Agent."""

from __future__ import annotations

from agent.comparison_verification import verify_comparison_state

from agent.comparison_reporting import write_comparison_reports

from agent.comparison import compare_two_image_results

from pathlib import Path
from typing import Any, Literal

from agent.reporting import save_single_image_report
from agent.state import IQAAgentState
from agent.verification import verify_single_image_state
from configs.config_loader import load_config
from models.cot_iqa_model import COTIQAModel
from rag.retriever import get_retriever
from tools.image_tools import inspect_image
from tools.pyiqa_tools import compare_images, evaluate_image


_CONFIG: dict[str, Any] | None = None
_COT_MODEL: COTIQAModel | None = None


def _get_config() -> dict[str, Any]:
    """Load and cache the project configuration."""

    global _CONFIG

    if _CONFIG is None:
        _CONFIG = load_config()

    return _CONFIG



def _get_cot_model() -> COTIQAModel:
    """Create and cache the CoT-IQA model wrapper."""

    global _COT_MODEL

    if _COT_MODEL is None:
        config = _get_config()
        _COT_MODEL = COTIQAModel(config["cot_iqa"])

    return _COT_MODEL


def unload_cot_iqa_model() -> None:
    """Unload the cached CoT-IQA model and release GPU memory."""

    global _COT_MODEL

    if _COT_MODEL is not None:
        _COT_MODEL.unload()
        _COT_MODEL = None


def _get_target_image_path(
    state: IQAAgentState,
) -> str | None:
    """Return the first image path for single-image analysis."""

    image_paths = state.get("image_paths", [])

    if not image_paths:
        return None

    image_path = str(image_paths[0]).strip()

    return image_path or None


def _resolve_project_path(
    path_value: str | Path,
) -> Path:
    """Resolve a configured or user-provided path."""

    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    config = _get_config()
    project_root = Path(
        config.get("_project_root", Path.cwd())
    ).expanduser()

    return (project_root / path).resolve()


def select_after_image_inspection(
    state: IQAAgentState,
) -> Literal["valid", "invalid"]:
    """Route valid images to IQA tools and invalid images to verification."""

    validation = (
        state.get("image_analysis", {})
        .get("validation", {})
    )

    return (
        "valid"
        if validation.get("valid", False)
        else "invalid"
    )


def inspect_single_image(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Validate and inspect the target image."""

    image_path = _get_target_image_path(state)

    if image_path is None:
        return {
            "image_metadata": {},
            "image_analysis": {
                "validation": {
                    "valid": False,
                    "error": "No image path was provided.",
                },
                "target_image": {},
            },
            "execution_trace": [
                "node:inspect_single_image",
            ],
            "errors": [
                "Image inspection failed: no image path was provided.",
            ],
        }

    resolved_path = _resolve_project_path(image_path)
    inspection = inspect_image(resolved_path)

    validation = inspection.get("validation", {})
    metadata = inspection.get("metadata", {})
    statistics = inspection.get("statistics", {})

    errors: list[str] = []

    if not validation.get("valid", False):
        errors.append(
            "Image inspection failed: "
            f"{validation.get('error') or 'unknown validation error'}"
        )

    return {
        "image_paths": [str(resolved_path)],
        "image_metadata": {
            "target_image": metadata,
        },
        "image_analysis": {
            "validation": validation,
            "target_image": statistics,
        },
        "execution_trace": [
            "node:inspect_single_image",
        ],
        "errors": errors,
    }


def run_pyiqa_single_image(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Run configured no-reference PyIQA metrics."""

    image_path = _get_target_image_path(state)

    if image_path is None:
        return {
            "pyiqa_results": {},
            "execution_trace": [
                "node:run_pyiqa_single_image",
            ],
            "errors": [
                "PyIQA failed: no image path was provided.",
            ],
        }

    validation = (
        state.get("image_analysis", {})
        .get("validation", {})
    )

    if not validation.get("valid", False):
        return {
            "pyiqa_results": {
                "target_image": {
                    "path": image_path,
                    "valid": False,
                    "device": None,
                    "results": {},
                    "errors": [],
                    "skipped": True,
                    "skip_reason": (
                        "PyIQA was skipped because the image is invalid."
                    ),
                }
            },
            "execution_trace": [
                "node:run_pyiqa_single_image",
            ],
            "errors": [],
        }

    config = _get_config()
    pyiqa_config = config.get("pyiqa", {})

    enabled = bool(
        pyiqa_config.get("enabled", True)
    )

    if not enabled:
        return {
            "pyiqa_results": {
                "target_image": {
                    "path": image_path,
                    "valid": True,
                    "device": None,
                    "results": {},
                    "errors": [],
                    "skipped": True,
                    "skip_reason": "PyIQA is disabled in configuration.",
                }
            },
            "execution_trace": [
                "node:run_pyiqa_single_image",
            ],
            "errors": [],
        }

    metrics = pyiqa_config.get(
        "metrics",
        ["niqe", "brisque"],
    )

    device = str(
        pyiqa_config.get("device", "auto")
    )

    result = evaluate_image(
        image_path=image_path,
        metrics=metrics,
        device=device,
    )

    node_errors = [
        f"PyIQA metric error: {error}"
        for error in result.get("errors", [])
    ]

    return {
        "pyiqa_results": {
            "target_image": result,
        },
        "execution_trace": [
            "node:run_pyiqa_single_image",
        ],
        "errors": node_errors,
    }

def run_cot_iqa_single_image(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Run Qwen2-VL + LoRA CoT-IQA inference."""

    image_path = _get_target_image_path(state)

    if image_path is None:
        return {
            "cot_iqa_raw_output": "",
            "cot_iqa_result": {
                "success": False,
                "error": "No image path was provided.",
            },
            "execution_trace": [
                "node:run_cot_iqa_single_image",
            ],
            "errors": [
                "CoT-IQA failed: no image path was provided.",
            ],
        }

    validation = (
        state.get("image_analysis", {})
        .get("validation", {})
    )

    if not validation.get("valid", False):
        return {
            "cot_iqa_raw_output": "",
            "cot_iqa_result": {
                "success": False,
                "image_path": image_path,
                "error": (
                    "CoT-IQA was skipped because the image is invalid."
                ),
                "skipped": True,
                "skip_reason": (
                    "Input image validation failed."
                ),
            },
            "execution_trace": [
                "node:run_cot_iqa_single_image",
            ],
            "errors": [],
        }

    try:
        model = _get_cot_model()

        result = model.analyze(
            image_path=image_path,
            prompt=None,
            keep_raw_output=True,
            auto_load=True,
        )

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "cot_iqa_raw_output": "",
            "cot_iqa_result": {
                "success": False,
                "image_path": image_path,
                "error": error_message,
            },
            "execution_trace": [
                "node:run_cot_iqa_single_image",
            ],
            "errors": [
                f"CoT-IQA loading or inference failed: {error_message}",
            ],
        }

    node_errors: list[str] = []

    if not result.get("success", False):
        node_errors.append(
            "CoT-IQA inference failed: "
            f"{result.get('error') or 'unknown error'}"
        )

    return {
        "cot_iqa_raw_output": result.get(
            "raw_output",
            "",
        ),
        "cot_iqa_result": result,
        "execution_trace": [
            "node:run_cot_iqa_single_image",
        ],
        "errors": node_errors,
    }

def verify_single_image_results(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Verify completeness and internal consistency of Agent outputs."""

    try:
        verification_result = verify_single_image_state(state)

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "verification_result": {
                "status": "failed",
                "checks": [],
                "check_count": 0,
                "passed_count": 0,
                "warning_count": 0,
                "failed_count": 1,
                "warnings": [],
                "errors": [
                    error_message,
                ],
                "summary": {},
            },
            "execution_trace": [
                "node:verify_single_image_results",
            ],
            "errors": [
                f"Verification failed: {error_message}",
            ],
        }

    return {
        "verification_result": verification_result,
        "execution_trace": [
            "node:verify_single_image_results",
        ],
        "errors": [],
    }

def generate_single_image_report(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Generate and persist JSON and Markdown reports."""

    try:
        report_result = save_single_image_report(state)

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "final_result": {
                "request_id": state.get("request_id"),
                "status": "failed",
                "task_type": state.get(
                    "task_type",
                    "unknown",
                ),
                "error": error_message,
            },
            "final_report": "",
            "execution_trace": [
                "node:generate_single_image_report",
            ],
            "errors": [
                f"Report generation failed: {error_message}",
            ],
        }

    return {
        "final_result": report_result["final_result"],
        "final_report": report_result["final_report"],
        "execution_trace": [
            "node:generate_single_image_report",
        ],
        "errors": [],
    }

def _build_comparison_items(
    state: IQAAgentState,
) -> list[dict[str, Any]]:
    """Build ordered comparison items from request image fields."""

    image_paths = [
        str(path).strip()
        for path in state.get("image_paths", [])
        if str(path).strip()
    ]

    reference_path = state.get("reference_image_path")

    has_reference = (
        isinstance(reference_path, str)
        and bool(reference_path.strip())
    )

    items: list[dict[str, Any]] = []

    if has_reference:
        for index, source_path in enumerate(
            image_paths,
            start=1,
        ):
            item_id = (
                "target"
                if len(image_paths) == 1
                else f"target_{index}"
            )

            items.append(
                {
                    "item_id": item_id,
                    "role": "target",
                    "source_path": source_path,
                }
            )

        items.append(
            {
                "item_id": "reference",
                "role": "reference",
                "source_path": reference_path.strip(),
            }
        )

    else:
        for index, source_path in enumerate(
            image_paths,
            start=1,
        ):
            items.append(
                {
                    "item_id": f"image_{index}",
                    "role": "candidate",
                    "source_path": source_path,
                }
            )

    return items


def inspect_comparison_images(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Validate and inspect all images in a comparison request."""

    source_items = _build_comparison_items(state)

    if len(source_items) < 2:
        return {
            "comparison_items": source_items,
            "comparison_image_metadata": {},
            "comparison_image_analysis": {
                "total_count": len(source_items),
                "valid_count": 0,
                "all_valid": False,
                "duplicate_paths": [],
                "items": {},
            },
            "execution_trace": [
                "node:inspect_comparison_images",
            ],
            "errors": [
                "Comparison inspection requires at least two images."
            ],
        }

    resolved_items: list[dict[str, Any]] = []
    metadata_by_item: dict[str, Any] = {}
    analysis_by_item: dict[str, Any] = {}
    node_errors: list[str] = []

    resolved_paths: list[str] = []

    for item in source_items:
        resolved_path = _resolve_project_path(
            item["source_path"]
        )

        resolved_path_text = str(resolved_path)
        resolved_paths.append(resolved_path_text)

        resolved_item = {
            **item,
            "resolved_path": resolved_path_text,
        }

        resolved_items.append(resolved_item)

        inspection = inspect_image(resolved_path)
        validation = inspection.get(
            "validation",
            {},
        )

        metadata_by_item[item["item_id"]] = (
            inspection.get("metadata", {})
        )

        analysis_by_item[item["item_id"]] = {
            "validation": validation,
            "statistics": inspection.get(
                "statistics",
                {},
            ),
        }

        if not validation.get("valid", False):
            node_errors.append(
                f"Comparison image {item['item_id']} "
                f"failed validation: "
                f"{validation.get('error') or 'unknown error'}"
            )

    duplicate_paths = sorted(
        {
            current_path
            for current_path in resolved_paths
            if resolved_paths.count(current_path) > 1
        }
    )

    if duplicate_paths:
        node_errors.append(
            "Comparison requires distinct image files; "
            f"duplicate paths found: {duplicate_paths}"
        )

    valid_count = sum(
        bool(
            item_analysis.get(
                "validation",
                {},
            ).get("valid", False)
        )
        for item_analysis in analysis_by_item.values()
    )

    all_valid = (
        valid_count == len(resolved_items)
        and not duplicate_paths
    )

    return {
        "comparison_items": resolved_items,
        "comparison_image_metadata": metadata_by_item,
        "comparison_image_analysis": {
            "total_count": len(resolved_items),
            "valid_count": valid_count,
            "all_valid": all_valid,
            "duplicate_paths": duplicate_paths,
            "items": analysis_by_item,
        },
        "execution_trace": [
            "node:inspect_comparison_images",
        ],
        "errors": node_errors,
    }


def select_after_comparison_inspection(
    state: IQAAgentState,
) -> Literal["valid", "invalid"]:
    """Route valid comparison inputs to IQA tools."""

    analysis = state.get(
        "comparison_image_analysis",
        {},
    )

    return (
        "valid"
        if analysis.get("all_valid", False)
        else "invalid"
    )


def run_pyiqa_comparison(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Evaluate and rank comparison images using PyIQA metrics."""

    items = state.get(
        "comparison_items",
        [],
    )

    comparison_analysis = state.get(
        "comparison_image_analysis",
        {},
    )

    if (
        len(items) < 2
        or not comparison_analysis.get(
            "all_valid",
            False,
        )
    ):
        return {
            "comparison_pyiqa_results": {
                "skipped": True,
                "skip_reason": (
                    "Comparison images are invalid "
                    "or fewer than two."
                ),
                "evaluations": {},
                "rankings": {},
                "pairwise": {},
            },
            "execution_trace": [
                "node:run_pyiqa_comparison",
            ],
            "errors": [],
        }

    config = _get_config()
    pyiqa_config = config.get("pyiqa", {})

    if not bool(
        pyiqa_config.get("enabled", True)
    ):
        return {
            "comparison_pyiqa_results": {
                "skipped": True,
                "skip_reason": (
                    "PyIQA is disabled in configuration."
                ),
                "evaluations": {},
                "rankings": {},
                "pairwise": {},
            },
            "execution_trace": [
                "node:run_pyiqa_comparison",
            ],
            "errors": [],
        }

    metrics = pyiqa_config.get(
        "metrics",
        ["niqe", "brisque"],
    )

    device = str(
        pyiqa_config.get("device", "auto")
    )

    resolved_paths = [
        item["resolved_path"]
        for item in items
    ]

    raw_result = compare_images(
        image_paths=resolved_paths,
        metrics=metrics,
        device=device,
    )

    item_by_path = {
        item["resolved_path"]: item
        for item in items
    }

    evaluations_by_item: dict[str, Any] = {}
    node_errors: list[str] = []

    for evaluated_path, evaluation in (
        raw_result.get("evaluations", {})
        .items()
    ):
        item = item_by_path.get(evaluated_path)

        if item is None:
            continue

        item_id = item["item_id"]

        evaluations_by_item[item_id] = {
            **evaluation,
            "item_id": item_id,
            "role": item["role"],
            "path": evaluated_path,
        }

        for error in evaluation.get(
            "errors",
            [],
        ):
            node_errors.append(
                f"PyIQA comparison {item_id}: {error}"
            )

    rankings: dict[str, list[dict[str, Any]]] = {}

    for metric_name, entries in (
        raw_result.get("rankings", {})
        .items()
    ):
        ranked_entries: list[dict[str, Any]] = []

        for rank, entry in enumerate(
            entries,
            start=1,
        ):
            item = item_by_path.get(
                entry["path"]
            )

            if item is None:
                continue

            ranked_entries.append(
                {
                    "rank": rank,
                    "item_id": item["item_id"],
                    "role": item["role"],
                    "path": entry["path"],
                    "score": entry["score"],
                }
            )

        rankings[metric_name] = ranked_entries

    pairwise: dict[str, Any] = {}

    if len(items) == 2:
        first_item = items[0]
        second_item = items[1]

        first_id = first_item["item_id"]
        second_id = second_item["item_id"]

        first_evaluation = evaluations_by_item.get(
            first_id,
            {},
        )

        second_evaluation = evaluations_by_item.get(
            second_id,
            {},
        )

        for metric_name in metrics:
            normalized_name = (
                str(metric_name)
                .strip()
                .lower()
            )

            first_metric = (
                first_evaluation.get(
                    "results",
                    {},
                ).get(normalized_name)
            )

            second_metric = (
                second_evaluation.get(
                    "results",
                    {},
                ).get(normalized_name)
            )

            if (
                first_metric is None
                or second_metric is None
            ):
                continue

            first_score = first_metric.get(
                "score"
            )

            second_score = second_metric.get(
                "score"
            )

            if not isinstance(
                first_score,
                float,
            ) or not isinstance(
                second_score,
                float,
            ):
                continue

            lower_better = bool(
                first_metric.get(
                    "lower_better",
                    False,
                )
            )

            if abs(
                first_score - second_score
            ) <= 1e-12:
                winner_item_id = None
                result_label = "tie"

            elif lower_better:
                winner_item_id = (
                    first_id
                    if first_score < second_score
                    else second_id
                )
                result_label = "winner"

            else:
                winner_item_id = (
                    first_id
                    if first_score > second_score
                    else second_id
                )
                result_label = "winner"

            quality_margin_second_over_first = (
                first_score - second_score
                if lower_better
                else second_score - first_score
            )

            pairwise[normalized_name] = {
                "first_item_id": first_id,
                "second_item_id": second_id,
                "first_score": first_score,
                "second_score": second_score,
                "lower_better": lower_better,
                "absolute_difference": round(
                    abs(
                        second_score
                        - first_score
                    ),
                    6,
                ),
                "quality_margin_second_over_first": round(
                    quality_margin_second_over_first,
                    6,
                ),
                "winner_item_id": winner_item_id,
                "result": result_label,
            }

    return {
        "comparison_pyiqa_results": {
            "skipped": False,
            "evaluations": evaluations_by_item,
            "rankings": rankings,
            "pairwise": pairwise,
        },
        "execution_trace": [
            "node:run_pyiqa_comparison",
        ],
        "errors": node_errors,
    }

def run_cot_iqa_comparison(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Run CoT-IQA sequentially for all comparison images."""

    items = state.get(
        "comparison_items",
        [],
    )

    comparison_analysis = state.get(
        "comparison_image_analysis",
        {},
    )

    if (
        len(items) < 2
        or not comparison_analysis.get(
            "all_valid",
            False,
        )
    ):
        return {
            "comparison_cot_results": {
                "skipped": True,
                "skip_reason": (
                    "Comparison images are invalid "
                    "or fewer than two."
                ),
                "item_count": len(items),
                "success_count": 0,
                "failed_count": 0,
                "items": {},
            },
            "execution_trace": [
                "node:run_cot_iqa_comparison",
            ],
            "errors": [],
        }

    try:
        model = _get_cot_model()

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "comparison_cot_results": {
                "skipped": False,
                "load_failed": True,
                "item_count": len(items),
                "success_count": 0,
                "failed_count": len(items),
                "items": {},
                "error": error_message,
            },
            "execution_trace": [
                "node:run_cot_iqa_comparison",
            ],
            "errors": [
                "Comparison CoT-IQA model loading failed: "
                f"{error_message}"
            ],
        }

    results_by_item: dict[str, Any] = {}
    node_errors: list[str] = []

    success_count = 0
    failed_count = 0

    for item in items:
        item_id = item["item_id"]
        role = item["role"]
        image_path = item.get(
            "resolved_path",
            item.get("source_path"),
        )

        try:
            result = model.analyze(
                image_path=image_path,
                prompt=None,
                keep_raw_output=True,
                auto_load=True,
            )

        except Exception as exc:
            error_message = (
                f"{type(exc).__name__}: {exc}"
            )

            result = {
                "success": False,
                "image_path": str(image_path),
                "raw_output": "",
                "parsed_output": None,
                "generated_token_count": 0,
                "hit_max_new_tokens": False,
                "ended_with_eos": False,
                "inference_time_seconds": 0.0,
                "error": error_message,
            }

        enriched_result = {
            **result,
            "item_id": item_id,
            "role": role,
        }

        results_by_item[item_id] = (
            enriched_result
        )

        if result.get("success", False):
            success_count += 1

        else:
            failed_count += 1

            node_errors.append(
                f"Comparison CoT-IQA {item_id} failed: "
                f"{result.get('error') or 'unknown error'}"
            )

    return {
        "comparison_cot_results": {
            "skipped": False,
            "load_failed": False,
            "item_count": len(items),
            "success_count": success_count,
            "failed_count": failed_count,
            "items": results_by_item,
        },
        "execution_trace": [
            "node:run_cot_iqa_comparison",
        ],
        "errors": node_errors,
    }

def compare_results(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Fuse PyIQA and CoT-IQA evidence for two-image comparison."""

    try:
        comparison_result = compare_two_image_results(
            state
        )

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        comparison_result = {
            "status": "failed",
            "winner_item_id": None,
            "loser_item_id": None,
            "confidence": "none",
            "decision_basis": "comparison_node_failure",
            "trusted_vote_counts": {},
            "trusted_vote_total": 0,
            "evidence": {},
            "mos_conflicts_with_primary": False,
            "conflicts": [],
            "warnings": [],
            "errors": [
                "Comparison decision failed: "
                f"{error_message}"
            ],
            "rationale": [],
        }

    node_errors = [
        f"Comparison decision error: {error}"
        for error in (
            comparison_result.get("errors")
            or []
        )
    ]

    return {
        "comparison_result": comparison_result,
        "execution_trace": [
            "node:compare_results",
        ],
        "errors": node_errors,
    }

def verify_comparison_results(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Verify the complete pairwise comparison workflow."""

    try:
        verification_result = (
            verify_comparison_state(state)
        )

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        verification_result = {
            "status": "failed",
            "checks": [],
            "check_count": 0,
            "passed_count": 0,
            "warning_count": 0,
            "failed_count": 1,
            "warnings": [],
            "errors": [
                "Comparison verification failed: "
                f"{error_message}"
            ],
            "summary": {
                "winner_item_id": None,
                "verification_exception": True,
            },
        }

    return {
        "verification_result": verification_result,
        "execution_trace": [
            "node:verify_comparison_results",
        ],
        "errors": [],
    }


def generate_comparison_report(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Generate JSON and Markdown comparison reports."""

    try:
        config = _get_config()

        configured_report_dir = (
            config.get("paths", {})
            .get(
                "report_dir",
                "outputs/reports",
            )
        )

        report_dir = _resolve_project_path(
            configured_report_dir
        )

        report_result = (
            write_comparison_reports(
                state=state,
                report_dir=report_dir,
            )
        )

        return {
            "final_result": report_result[
                "final_result"
            ],
            "final_report": report_result[
                "final_report"
            ],
            "execution_trace": [
                "node:generate_comparison_report",
            ],
            "errors": [],
        }

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        failed_result = {
            "request_id": state.get(
                "request_id"
            ),
            "task_type": (
                "multi_image_comparison"
            ),
            "status": "failed",
            "winner_item_id": (
                state.get(
                    "comparison_result",
                    {},
                ).get("winner_item_id")
            ),
            "loser_item_id": (
                state.get(
                    "comparison_result",
                    {},
                ).get("loser_item_id")
            ),
            "confidence": (
                state.get(
                    "comparison_result",
                    {},
                ).get(
                    "confidence",
                    "none",
                )
            ),
            "report_json_path": None,
            "report_markdown_path": None,
            "error": error_message,
        }

        return {
            "final_result": failed_result,
            "final_report": "",
            "execution_trace": [
                "node:generate_comparison_report",
            ],
            "errors": [
                "Comparison report generation failed: "
                f"{error_message}"
            ],
        }

def _flatten_rag_value(
    value: Any,
    output: list[str],
    *,
    depth: int = 0,
) -> None:
    """Extract concise semantic text from nested workflow results."""

    if depth > 4 or len(output) >= 50:
        return

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()

            if any(
                ignored in key_lower
                for ignored in (
                    "raw_output",
                    "image_path",
                    "request_id",
                    "generation_time",
                    "token_count",
                    "trace",
                    "error",
                )
            ):
                continue

            if isinstance(item, (str, int, float, bool)):
                item_text = str(item).strip()

                if item_text:
                    output.append(
                        f"{key_text}: {item_text}"
                    )
            else:
                _flatten_rag_value(
                    item,
                    output,
                    depth=depth + 1,
                )

    elif isinstance(value, (list, tuple)):
        for item in value[:10]:
            _flatten_rag_value(
                item,
                output,
                depth=depth + 1,
            )

    elif isinstance(value, (str, int, float, bool)):
        item_text = str(value).strip()

        if item_text:
            output.append(item_text)


def _build_rag_query(
    state: IQAAgentState,
) -> str:
    """Build a retrieval query from the current Agent evidence."""

    route = str(
        state.get("route", "")
    ).lower()

    user_query = str(
        state.get("user_query", "")
    ).strip()

    query_parts: list[str] = []

    if user_query:
        query_parts.append(user_query)

    if "comparison" in route:
        query_parts.append(
            "paired image quality comparison, "
            "distortion diagnosis, perceptual quality "
            "reasoning and restoration guidance"
        )

        _flatten_rag_value(
            state.get("comparison_result", {}),
            query_parts,
        )

        _flatten_rag_value(
            state.get("comparison_cot_results", {}),
            query_parts,
        )

    else:
        query_parts.append(
            "single image quality assessment, "
            "visual degradation diagnosis, severity, "
            "localization and restoration guidance"
        )

        _flatten_rag_value(
            state.get("cot_iqa_result", {}),
            query_parts,
        )

    query = " | ".join(query_parts)

    return query[:1800]


def retrieve_knowledge(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Retrieve supporting IQA knowledge for either workflow branch."""

    node_name = "retrieve_knowledge"

    try:
        config = _get_config()
        rag_config = config.get("rag", {})

        top_k = int(
            rag_config.get("top_k", 4)
        )

        query = _build_rag_query(state)

        runtime_device = str(
            rag_config.get(
                "runtime_device",
                "cpu",
            )
        ).strip() or "cpu"

        retriever = get_retriever(
            device=runtime_device,
        )

        results = retriever.search(
            query,
            top_k=top_k,
            deduplicate_pages=True,
        )

        return {
            "rag_context": results,
            "execution_trace": [node_name],
        }

    except Exception as exc:
        return {
            "rag_context": [],
            "errors": [
                {
                    "node": node_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
            "execution_trace": [node_name],
        }

