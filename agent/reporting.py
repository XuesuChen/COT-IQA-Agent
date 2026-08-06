"""Report generation and persistence for COT-IQA-Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from configs.config_loader import load_config


def _get_report_dir() -> Path:
    """Resolve and create the configured report directory."""

    config = load_config()
    project_root = Path(
        config.get("_project_root", Path.cwd())
    ).expanduser()

    report_dir = Path(
        config.get("paths", {}).get(
            "report_dir",
            "outputs/reports",
        )
    ).expanduser()

    if not report_dir.is_absolute():
        report_dir = project_root / report_dir

    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    return report_dir


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """Write text atomically to avoid incomplete report files."""

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        content,
        encoding="utf-8",
    )

    temporary_path.replace(path)


def _get_target_image_path(
    state: Mapping[str, Any],
) -> str | None:
    """Return the first image path from the workflow state."""

    image_paths = state.get("image_paths", []) or []

    if not image_paths:
        return None

    return str(image_paths[0])


def _build_final_result(
    state: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    """Build the compact result returned to API/UI callers."""

    verification = (
        state.get("verification_result", {})
        or {}
    )

    cot_result = (
        state.get("cot_iqa_result", {})
        or {}
    )

    parsed = (
        cot_result.get("parsed_output")
        or {}
    )

    quality = (
        parsed.get("quality_prediction")
        or {}
    )

    expert_routing = (
        parsed.get("expert_routing")
        or {}
    )

    pyiqa_result = (
        (state.get("pyiqa_results", {}) or {})
        .get("target_image", {})
        or {}
    )

    metric_results = (
        pyiqa_result.get("results")
        or {}
    )

    metric_scores = {
        metric_name: {
            "score": metric_result.get("score"),
            "lower_better": metric_result.get(
                "lower_better"
            ),
        }
        for metric_name, metric_result
        in metric_results.items()
    }

    return {
        "request_id": state.get("request_id"),
        "status": verification.get(
            "status",
            "unknown",
        ),
        "task_type": state.get(
            "task_type",
            "unknown",
        ),
        "image_path": _get_target_image_path(
            state
        ),
        "predicted_mos": quality.get(
            "predicted_mos"
        ),
        "mos_scale": quality.get(
            "mos_scale"
        ),
        "normalized_mos": quality.get(
            "normalized_mos"
        ),
        "selected_expert": expert_routing.get(
            "selected_expert"
        ),
        "top_experts": expert_routing.get(
            "top_experts",
            [],
        ),
        "pyiqa_scores": metric_scores,
        "verification": {
            "check_count": verification.get(
                "check_count",
                0,
            ),
            "passed_count": verification.get(
                "passed_count",
                0,
            ),
            "warning_count": verification.get(
                "warning_count",
                0,
            ),
            "failed_count": verification.get(
                "failed_count",
                0,
            ),
        },
        "report_json_path": str(json_path),
        "report_markdown_path": str(
            markdown_path
        ),
    }


def build_single_image_markdown(
    state: Mapping[str, Any],
) -> str:
    """Build a readable Markdown report for single-image IQA."""

    request_id = state.get(
        "request_id",
        "unknown",
    )

    user_query = state.get(
        "user_query",
        "",
    )

    image_path = (
        _get_target_image_path(state)
        or "N/A"
    )

    metadata = (
        (state.get("image_metadata", {}) or {})
        .get("target_image", {})
        or {}
    )

    image_analysis = (
        state.get("image_analysis", {})
        or {}
    )

    validation = (
        image_analysis.get("validation")
        or {}
    )

    statistics = (
        image_analysis.get("target_image")
        or {}
    )

    pyiqa_result = (
        (state.get("pyiqa_results", {}) or {})
        .get("target_image", {})
        or {}
    )

    cot_result = (
        state.get("cot_iqa_result", {})
        or {}
    )

    parsed = (
        cot_result.get("parsed_output")
        or {}
    )

    verification = (
        state.get("verification_result", {})
        or {}
    )

    localization = (
        parsed.get("localization")
        or []
    )

    attribution = (
        parsed.get("attribution")
        or {}
    )

    diagnosis = (
        parsed.get("diagnosis")
        or {}
    )

    restoration = (
        parsed.get("restoration_suggestion")
        or []
    )

    expert_routing = (
        parsed.get("expert_routing")
        or {}
    )

    quality = (
        parsed.get("quality_prediction")
        or {}
    )

    lines: list[str] = []

    lines.extend(
        [
            "# COT-IQA-Agent Image Quality Report",
            "",
            "## 1. Request Summary",
            "",
            f"- **Request ID:** `{request_id}`",
            f"- **Task Type:** `{state.get('task_type', 'unknown')}`",
            f"- **Workflow Status:** `{verification.get('status', 'unknown')}`",
            f"- **User Query:** {user_query}",
            f"- **Image Path:** `{image_path}`",
            "",
            "## 2. Image Information",
            "",
            f"- **Valid:** {validation.get('valid', False)}",
            f"- **Format:** {metadata.get('format', validation.get('format', 'N/A'))}",
            f"- **Mode:** {metadata.get('mode', validation.get('mode', 'N/A'))}",
            (
                "- **Dimensions:** "
                f"{metadata.get('width', validation.get('width', 'N/A'))}"
                " × "
                f"{metadata.get('height', validation.get('height', 'N/A'))}"
            ),
            f"- **Megapixels:** {metadata.get('megapixels', 'N/A')}",
            f"- **File Size:** {metadata.get('file_size_kb', 'N/A')} KB",
            "",
        ]
    )

    if validation.get("error"):
        lines.append(
            f"- **Validation Error:** {validation['error']}"
        )
        lines.append("")

    lines.extend(
        [
            "## 3. Basic Image Statistics",
            "",
        ]
    )

    if statistics:
        for key, value in statistics.items():
            lines.append(
                f"- **{key}:** {value}"
            )
    else:
        lines.append(
            "- No image statistics are available."
        )

    lines.extend(
        [
            "",
            "## 4. PyIQA Results",
            "",
        ]
    )

    metric_results = (
        pyiqa_result.get("results")
        or {}
    )

    if metric_results:
        lines.extend(
            [
                "| Metric | Score | Direction |",
                "|---|---:|---|",
            ]
        )

        for metric_name, metric_result in metric_results.items():
            direction = (
                "Lower is better"
                if metric_result.get("lower_better")
                else "Higher is better"
            )

            lines.append(
                f"| {metric_name.upper()} "
                f"| {metric_result.get('score')} "
                f"| {direction} |"
            )
    else:
        lines.append(
            "- No PyIQA scores are available."
        )

    for error in pyiqa_result.get("errors", []) or []:
        lines.append(
            f"- **PyIQA Warning:** {error}"
        )

    lines.extend(
        [
            "",
            "## 5. CoT-IQA Inference",
            "",
            f"- **Success:** {cot_result.get('success', False)}",
            (
                "- **Generated Tokens:** "
                f"{cot_result.get('generated_token_count', 0)}"
            ),
            (
                "- **Generation Time:** "
                f"{cot_result.get('inference_time_seconds', 0.0)} seconds"
            ),
            (
                "- **Reached Token Limit:** "
                f"{cot_result.get('hit_max_new_tokens', False)}"
            ),
            (
                "- **Ended With EOS:** "
                f"{cot_result.get('ended_with_eos', False)}"
            ),
            (
                "- **Six-Section Complete:** "
                f"{parsed.get('complete', False)}"
            ),
            "",
            "### 5.1 Degradation Localization",
            "",
        ]
    )

    if localization:
        lines.extend(
            [
                "| Region | Bounding Box | Distortion | Severity | Scope |",
                "|---|---|---|---|---|",
            ]
        )

        for item in localization:
            lines.append(
                f"| {item.get('region')} "
                f"| `{item.get('bbox')}` "
                f"| {item.get('distortion_raw', item.get('distortion'))} "
                f"| {item.get('severity')} "
                f"| {item.get('scope')} |"
            )
    else:
        lines.append(
            "- No degradation regions were parsed."
        )

    lines.extend(
        [
            "",
            "### 5.2 Degradation Attribution",
            "",
            (
                "- **Degradation Count:** "
                f"{attribution.get('degradation_count')}"
            ),
            (
                "- **Primary Impairment:** "
                f"{attribution.get('primary_impairment_raw', attribution.get('primary_impairment'))}"
            ),
            (
                "- **Attribution Text:** "
                f"{attribution.get('text', 'N/A')}"
            ),
            "",
            "### 5.3 Quality Dimension Diagnosis",
            "",
        ]
    )

    diagnosis_rows = (
        diagnosis.get("rows")
        or []
    )

    if diagnosis_rows:
        lines.extend(
            [
                "| Dimension | Score | Description |",
                "|---|---:|---|",
            ]
        )

        for row in diagnosis_rows:
            lines.append(
                f"| {row.get('dimension_raw', row.get('dimension'))} "
                f"| {row.get('score')} "
                f"| {row.get('description', '')} |"
            )

        lines.append("")
        lines.append(
            f"- **Calculated Mean Score:** {diagnosis.get('mean_score')}"
        )
    else:
        lines.append(
            "- No diagnosis dimensions were parsed."
        )

    lines.extend(
        [
            "",
            "### 5.4 Restoration Suggestions",
            "",
        ]
    )

    if restoration:
        for item in restoration:
            lines.append(
                f"- **Region {item.get('region')} — "
                f"{item.get('distortion_raw', item.get('distortion'))}:** "
                f"{item.get('action')}"
            )
    else:
        lines.append(
            "- No restoration suggestions were parsed."
        )

    lines.extend(
        [
            "",
            "### 5.5 Expert Routing",
            "",
            (
                "- **Selected Expert:** "
                f"{expert_routing.get('selected_expert')}"
            ),
            (
                "- **Top Experts:** "
                f"{expert_routing.get('top_experts', [])}"
            ),
            (
                "- **Selection Ambiguous:** "
                f"{expert_routing.get('selection_ambiguous')}"
            ),
            (
                "- **Raw Weight Sum:** "
                f"{expert_routing.get('weight_sum')}"
            ),
            (
                "- **Normalization Status:** "
                f"{expert_routing.get('normalization_status')}"
            ),
            "",
        ]
    )

    normalized_weights = (
        expert_routing.get(
            "normalized_weights"
        )
        or {}
    )

    if normalized_weights:
        lines.extend(
            [
                "| Expert | Normalized Weight |",
                "|---|---:|",
            ]
        )

        for expert_name, weight in normalized_weights.items():
            lines.append(
                f"| {expert_name} | {weight} |"
            )

    lines.extend(
        [
            "",
            "### 5.6 Overall Quality Prediction",
            "",
            (
                "- **Predicted MOS:** "
                f"{quality.get('predicted_mos')} / "
                f"{quality.get('mos_scale')}"
            ),
            (
                "- **Normalized MOS:** "
                f"{quality.get('normalized_mos')}"
            ),
            (
                "- **Within Declared Range:** "
                f"{quality.get('in_range')}"
            ),
            (
                "- **Reasoning:** "
                f"{quality.get('reasoning', 'N/A')}"
            ),
            "",
            "## 6. Verification",
            "",
            (
                "- **Overall Status:** "
                f"`{verification.get('status', 'unknown')}`"
            ),
            (
                "- **Checks:** "
                f"{verification.get('check_count', 0)}"
            ),
            (
                "- **Passed:** "
                f"{verification.get('passed_count', 0)}"
            ),
            (
                "- **Warnings:** "
                f"{verification.get('warning_count', 0)}"
            ),
            (
                "- **Failed:** "
                f"{verification.get('failed_count', 0)}"
            ),
            "",
        ]
    )

    checks = (
        verification.get("checks")
        or []
    )

    if checks:
        lines.extend(
            [
                "| Check | Status | Message |",
                "|---|---|---|",
            ]
        )

        for check in checks:
            lines.append(
                f"| {check.get('name')} "
                f"| {check.get('status')} "
                f"| {check.get('message')} |"
            )

    verification_warnings = (
        verification.get("warnings")
        or []
    )

    if verification_warnings:
        lines.extend(
            [
                "",
                "### Verification Warnings",
                "",
            ]
        )

        for warning in verification_warnings:
            lines.append(
                f"- {warning}"
            )

    state_errors = state.get("errors", []) or []

    if state_errors:
        lines.extend(
            [
                "",
                "## 7. Workflow Errors",
                "",
            ]
        )

        for error in state_errors:
            lines.append(
                f"- {error}"
            )

    lines.extend(
        [
            "",
            "## 8. Execution Trace",
            "",
        ]
    )

    for trace_item in state.get(
        "execution_trace",
        [],
    ) or []:
        lines.append(
            f"- `{trace_item}`"
        )

    raw_output = str(
        state.get(
            "cot_iqa_raw_output",
            cot_result.get("raw_output", ""),
        )
        or ""
    )

    lines.extend(
        [
            "",
            "## 9. Raw CoT-IQA Output",
            "",
            "<details>",
            "<summary>Show raw model output</summary>",
            "",
            "```text",
            raw_output,
            "```",
            "",
            "</details>",
            "",
        ]
    )

    return "\n".join(lines)


def save_single_image_report(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Save JSON and Markdown reports and return final output data."""

    request_id = str(
        state.get("request_id", "unknown")
    )

    report_dir = _get_report_dir()

    json_path = (
        report_dir
        / f"{request_id}_single_image.json"
    )

    markdown_path = (
        report_dir
        / f"{request_id}_single_image.md"
    )

    final_result = _build_final_result(
        state=state,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    markdown_report = (
        build_single_image_markdown(state)
    )

    payload = {
        "request_id": state.get("request_id"),
        "user_query": state.get("user_query"),
        "task_type": state.get("task_type"),
        "route": state.get("route"),
        "plan": state.get("plan", []),
        "image_paths": state.get(
            "image_paths",
            [],
        ),
        "reference_image_path": state.get(
            "reference_image_path"
        ),
        "image_metadata": state.get(
            "image_metadata",
            {},
        ),
        "image_analysis": state.get(
            "image_analysis",
            {},
        ),
        "pyiqa_results": state.get(
            "pyiqa_results",
            {},
        ),
        "cot_iqa_raw_output": state.get(
            "cot_iqa_raw_output",
            "",
        ),
        "cot_iqa_result": state.get(
            "cot_iqa_result",
            {},
        ),
        "rag_context": state.get(
            "rag_context",
            [],
        ),
        "verification_result": state.get(
            "verification_result",
            {},
        ),
        "final_result": final_result,
        "execution_trace": state.get(
            "execution_trace",
            [],
        ),
        "errors": state.get(
            "errors",
            [],
        ),
    }

    json_content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    _atomic_write_text(
        json_path,
        json_content + "\n",
    )

    _atomic_write_text(
        markdown_path,
        markdown_report,
    )

    return {
        "final_result": final_result,
        "final_report": markdown_report,
        "report_json_path": str(json_path),
        "report_markdown_path": str(
            markdown_path
        ),
    }
