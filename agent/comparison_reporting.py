"""JSON and Markdown reporting for image comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


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
    payload: Mapping[str, Any],
) -> None:
    """Atomically write formatted JSON."""

    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    _atomic_write_text(
        path,
        content + "\n",
    )


def _escape_markdown(value: Any) -> str:
    """Escape a value for Markdown table cells."""

    if value is None:
        return "—"

    return (
        str(value)
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def _format_number(value: Any) -> str:
    """Format numeric report values."""

    if isinstance(value, float):
        return f"{value:.4f}"

    if isinstance(value, int):
        return str(value)

    return _escape_markdown(value)


def _get_cot_summary(
    state: Mapping[str, Any],
    item_id: str,
) -> dict[str, Any]:
    """Extract compact CoT information for one item."""

    cot_data = (
        state.get(
            "comparison_cot_results",
            {},
        )
        or {}
    )

    item_result = (
        cot_data.get("items", {})
        .get(item_id, {})
        or {}
    )

    parsed = (
        item_result.get("parsed_output")
        or {}
    )

    diagnosis = (
        parsed.get("diagnosis")
        or {}
    )

    routing = (
        parsed.get("expert_routing")
        or {}
    )

    quality = (
        parsed.get("quality_prediction")
        or {}
    )

    return {
        "success": bool(
            item_result.get("success", False)
        ),
        "complete": bool(
            parsed.get("complete", False)
        ),
        "diagnosis_mean": diagnosis.get(
            "mean_score"
        ),
        "predicted_mos": quality.get(
            "predicted_mos"
        ),
        "mos_scale": quality.get(
            "mos_scale"
        ),
        "selected_expert": routing.get(
            "selected_expert"
        ),
        "top_experts": routing.get(
            "top_experts",
            [],
        ),
        "localization": parsed.get(
            "localization",
            [],
        ),
        "parser_warnings": parsed.get(
            "warnings",
            [],
        ),
        "raw_output": item_result.get(
            "raw_output",
            "",
        ),
        "error": item_result.get("error"),
    }


def build_comparison_markdown(
    state: Mapping[str, Any],
) -> str:
    """Build the human-readable comparison report."""

    request_id = state.get(
        "request_id",
        "unknown_request",
    )

    user_query = state.get(
        "user_query",
        "",
    )

    items = list(
        state.get("comparison_items", [])
        or []
    )

    metadata = (
        state.get(
            "comparison_image_metadata",
            {},
        )
        or {}
    )

    pyiqa = (
        state.get(
            "comparison_pyiqa_results",
            {},
        )
        or {}
    )

    decision = (
        state.get("comparison_result", {})
        or {}
    )

    verification = (
        state.get(
            "verification_result",
            {},
        )
        or {}
    )

    winner_item_id = decision.get(
        "winner_item_id"
    )

    loser_item_id = decision.get(
        "loser_item_id"
    )

    lines = [
        "# COT-IQA-Agent Image Comparison Report",
        "",
        "## Request",
        "",
        f"- **Request ID:** `{request_id}`",
        f"- **User query:** {_escape_markdown(user_query)}",
        f"- **Workflow status:** `{verification.get('status', decision.get('status', 'unknown'))}`",
        "",
        "## Final Decision",
        "",
        f"- **Better-quality image:** `{winner_item_id or 'inconclusive'}`",
        f"- **Lower-quality image:** `{loser_item_id or 'inconclusive'}`",
        f"- **Confidence:** `{decision.get('confidence', 'none')}`",
        f"- **Decision basis:** `{decision.get('decision_basis', 'unknown')}`",
        f"- **Primary evidence votes:** `{decision.get('trusted_vote_counts', {})}`",
        f"- **Generated MOS conflict:** `{decision.get('mos_conflicts_with_primary', False)}`",
        "",
        "## Input Images",
        "",
        "| Item | Role | Path | Resolution | Format |",
        "|---|---|---|---:|---|",
    ]

    for item in items:
        item_id = str(
            item.get("item_id")
        )

        item_metadata = (
            metadata.get(item_id, {})
            or {}
        )

        width = item_metadata.get(
            "width",
            "—",
        )

        height = item_metadata.get(
            "height",
            "—",
        )

        resolution = (
            f"{width} × {height}"
            if width != "—" and height != "—"
            else "—"
        )

        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(item_id),
                    _escape_markdown(
                        item.get("role")
                    ),
                    _escape_markdown(
                        item.get(
                            "resolved_path",
                            item.get("source_path"),
                        )
                    ),
                    _escape_markdown(resolution),
                    _escape_markdown(
                        item_metadata.get("format")
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## PyIQA Comparison",
            "",
            "| Metric | First score | Second score | Lower is better | Winner |",
            "|---|---:|---:|:---:|---|",
        ]
    )

    pairwise = (
        pyiqa.get("pairwise", {})
        or {}
    )

    if pairwise:
        for metric_name, metric_result in pairwise.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(
                            metric_name.upper()
                        ),
                        _format_number(
                            metric_result.get(
                                "first_score"
                            )
                        ),
                        _format_number(
                            metric_result.get(
                                "second_score"
                            )
                        ),
                        _escape_markdown(
                            metric_result.get(
                                "lower_better"
                            )
                        ),
                        _escape_markdown(
                            metric_result.get(
                                "winner_item_id"
                            )
                        ),
                    ]
                )
                + " |"
            )

    else:
        lines.append(
            "| — | — | — | — | No usable metric evidence |"
        )

    lines.extend(
        [
            "",
            "## CoT-IQA Comparison",
            "",
            "| Item | Success | Complete | Diagnosis mean ↓ | Predicted MOS ↑ | Selected expert |",
            "|---|:---:|:---:|---:|---:|---|",
        ]
    )

    cot_summaries: dict[str, dict[str, Any]] = {}

    for item in items:
        item_id = str(
            item.get("item_id")
        )

        summary = _get_cot_summary(
            state,
            item_id,
        )

        cot_summaries[item_id] = summary

        selected_expert = (
            summary.get("selected_expert")
            or ", ".join(
                summary.get("top_experts", [])
            )
            or "—"
        )

        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(item_id),
                    _escape_markdown(
                        summary["success"]
                    ),
                    _escape_markdown(
                        summary["complete"]
                    ),
                    _format_number(
                        summary["diagnosis_mean"]
                    ),
                    _format_number(
                        summary["predicted_mos"]
                    ),
                    _escape_markdown(
                        selected_expert
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Detected Degradations",
            "",
        ]
    )

    for item in items:
        item_id = str(
            item.get("item_id")
        )

        summary = cot_summaries[item_id]

        lines.append(
            f"### {item_id}"
        )

        localization = (
            summary.get("localization", [])
            or []
        )

        if not localization:
            lines.append(
                ""
            )
            lines.append(
                "- No degradation region was parsed."
            )
            lines.append(
                ""
            )
            continue

        lines.append("")
        lines.append(
            "| Region | Distortion | Severity | Scope | Bounding box |"
        )
        lines.append(
            "|---|---|---|---|---|"
        )

        for region in localization:
            distortion = (
                region.get("distortion_raw")
                or region.get("distortion")
            )

            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(
                            region.get("region")
                        ),
                        _escape_markdown(
                            distortion
                        ),
                        _escape_markdown(
                            region.get("severity")
                        ),
                        _escape_markdown(
                            region.get("scope")
                        ),
                        _escape_markdown(
                            region.get("bbox")
                        ),
                    ]
                )
                + " |"
            )

        lines.append("")

    lines.extend(
        [
            "## Decision Rationale",
            "",
        ]
    )

    rationale = (
        decision.get("rationale", [])
        or []
    )

    if rationale:
        lines.extend(
            f"- {_escape_markdown(item)}"
            for item in rationale
        )
    else:
        lines.append(
            "- No comparison rationale was produced."
        )

    lines.extend(
        [
            "",
            "## Evidence Conflicts",
            "",
        ]
    )

    conflicts = (
        decision.get("conflicts", [])
        or []
    )

    if conflicts:
        lines.extend(
            f"- {_escape_markdown(item)}"
            for item in conflicts
        )
    else:
        lines.append(
            "- No evidence conflicts were reported."
        )

    lines.extend(
        [
            "",
            "## Verification",
            "",
            "| Check | Status | Message |",
            "|---|---|---|",
        ]
    )

    checks = (
        verification.get("checks", [])
        or []
    )

    for check in checks:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(
                        check.get("name")
                    ),
                    _escape_markdown(
                        check.get("status")
                    ),
                    _escape_markdown(
                        check.get("message")
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Workflow Errors",
            "",
        ]
    )

    workflow_errors = list(
        state.get("errors", [])
        or []
    )

    if workflow_errors:
        lines.extend(
            f"- {_escape_markdown(item)}"
            for item in workflow_errors
        )
    else:
        lines.append(
            "- None."
        )

    lines.extend(
        [
            "",
            "## Execution Trace",
            "",
        ]
    )

    for trace_item in (
        state.get("execution_trace", [])
        or []
    ):
        lines.append(
            f"- `{_escape_markdown(trace_item)}`"
        )

    lines.extend(
        [
            "",
            "## Raw CoT-IQA Outputs",
            "",
        ]
    )

    for item in items:
        item_id = str(
            item.get("item_id")
        )

        raw_output = (
            cot_summaries[item_id]
            .get("raw_output")
            or ""
        )

        lines.extend(
            [
                f"<details>",
                f"<summary>{_escape_markdown(item_id)}</summary>",
                "",
                "```text",
                raw_output,
                "```",
                "",
                "</details>",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_comparison_reports(
    state: Mapping[str, Any],
    report_dir: str | Path,
) -> dict[str, Any]:
    """Write comparison JSON and Markdown reports."""

    request_id = str(
        state.get(
            "request_id",
            "unknown_request",
        )
    )

    output_dir = Path(
        report_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output_dir / (
        f"{request_id}_comparison.json"
    )

    markdown_path = output_dir / (
        f"{request_id}_comparison.md"
    )

    decision = (
        state.get("comparison_result", {})
        or {}
    )

    verification = (
        state.get(
            "verification_result",
            {},
        )
        or {}
    )

    final_result = {
        "request_id": request_id,
        "task_type": "multi_image_comparison",
        "status": verification.get(
            "status",
            decision.get("status", "unknown"),
        ),
        "winner_item_id": decision.get(
            "winner_item_id"
        ),
        "loser_item_id": decision.get(
            "loser_item_id"
        ),
        "confidence": decision.get(
            "confidence",
            "none",
        ),
        "decision_basis": decision.get(
            "decision_basis"
        ),
        "trusted_vote_counts": decision.get(
            "trusted_vote_counts",
            {},
        ),
        "mos_conflicts_with_primary": decision.get(
            "mos_conflicts_with_primary",
            False,
        ),
        "report_json_path": str(json_path),
        "report_markdown_path": str(
            markdown_path
        ),
    }

    markdown_report = build_comparison_markdown(
        state
    )

    payload = {
        "request": {
            "request_id": request_id,
            "user_query": state.get(
                "user_query"
            ),
            "image_paths": state.get(
                "image_paths",
                [],
            ),
            "reference_image_path": state.get(
                "reference_image_path"
            ),
        },
        "comparison_items": state.get(
            "comparison_items",
            [],
        ),
        "comparison_image_metadata": state.get(
            "comparison_image_metadata",
            {},
        ),
        "comparison_image_analysis": state.get(
            "comparison_image_analysis",
            {},
        ),
        "comparison_pyiqa_results": state.get(
            "comparison_pyiqa_results",
            {},
        ),
        "comparison_cot_results": state.get(
            "comparison_cot_results",
            {},
        ),
        "comparison_result": decision,
        "verification_result": verification,
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

    _atomic_write_json(
        json_path,
        payload,
    )

    _atomic_write_text(
        markdown_path,
        markdown_report,
    )

    return {
        "final_result": final_result,
        "final_report": markdown_report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
