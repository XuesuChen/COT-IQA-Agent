"""Verification utilities for COT-IQA-Agent outputs."""

from __future__ import annotations

import math
import re
from statistics import fmean
from typing import Any, Mapping


_REASONING_MEAN_RE = re.compile(
    r"mean\s+diagnosis\s+score\s+is\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
    re.IGNORECASE,
)


def _record_check(
    checks: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    *,
    name: str,
    status: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Append one structured verification check."""

    check: dict[str, Any] = {
        "name": name,
        "status": status,
        "message": message,
    }

    if details:
        check["details"] = details

    checks.append(check)

    if status == "warning":
        warnings.append(message)
    elif status == "failed":
        errors.append(message)


def _finite_number(value: Any) -> bool:
    """Return whether a value is a finite int or float."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def verify_single_image_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one completed single-image analysis state."""

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    image_analysis = state.get("image_analysis", {}) or {}
    validation = image_analysis.get("validation", {}) or {}

    pyiqa_result = (
        (state.get("pyiqa_results", {}) or {})
        .get("target_image", {})
        or {}
    )

    cot_result = state.get("cot_iqa_result", {}) or {}
    parsed = cot_result.get("parsed_output") or {}

    # 1. Image validity
    image_valid = bool(validation.get("valid", False))

    _record_check(
        checks,
        warnings,
        errors,
        name="image_validation",
        status="passed" if image_valid else "failed",
        message=(
            "Input image validation passed."
            if image_valid
            else (
                "Input image validation failed: "
                f"{validation.get('error') or 'unknown error'}"
            )
        ),
        details={
            "path": validation.get("path"),
            "format": validation.get("format"),
            "width": validation.get("width"),
            "height": validation.get("height"),
        },
    )

    if not image_valid:
        return {
            "status": "failed",
            "checks": checks,
            "check_count": len(checks),
            "passed_count": sum(
                check["status"] == "passed"
                for check in checks
            ),
            "warning_count": sum(
                check["status"] == "warning"
                for check in checks
            ),
            "failed_count": sum(
                check["status"] == "failed"
                for check in checks
            ),
            "warnings": warnings,
            "errors": errors,
            "summary": {
                "image_valid": False,
                "pyiqa_metrics": [],
                "cot_success": False,
                "cot_complete": False,
                "predicted_mos": None,
                "selected_expert": None,
                "downstream_processing_skipped": True,
            },
        }

    # 2. PyIQA execution
    pyiqa_scores = pyiqa_result.get("results", {}) or {}
    pyiqa_errors = pyiqa_result.get("errors", []) or []
    pyiqa_skipped = bool(pyiqa_result.get("skipped", False))

    if pyiqa_skipped:
        _record_check(
            checks,
            warnings,
            errors,
            name="pyiqa_execution",
            status="warning",
            message=(
                "PyIQA evaluation was skipped: "
                f"{pyiqa_result.get('skip_reason') or 'unspecified reason'}"
            ),
        )

    elif pyiqa_errors:
        _record_check(
            checks,
            warnings,
            errors,
            name="pyiqa_execution",
            status="warning",
            message=(
                f"PyIQA completed with {len(pyiqa_errors)} metric error(s)."
            ),
            details={
                "metric_names": sorted(pyiqa_scores),
                "metric_errors": pyiqa_errors,
            },
        )

    elif pyiqa_scores:
        _record_check(
            checks,
            warnings,
            errors,
            name="pyiqa_execution",
            status="passed",
            message="PyIQA metrics completed successfully.",
            details={
                "device": pyiqa_result.get("device"),
                "metric_names": sorted(pyiqa_scores),
            },
        )

    else:
        _record_check(
            checks,
            warnings,
            errors,
            name="pyiqa_execution",
            status="warning",
            message="No PyIQA metric result is available.",
        )

    # 3. CoT-IQA inference
    cot_success = bool(cot_result.get("success", False))
    raw_output = str(cot_result.get("raw_output", "") or "")

    _record_check(
        checks,
        warnings,
        errors,
        name="cot_iqa_inference",
        status="passed" if cot_success and raw_output else "failed",
        message=(
            "CoT-IQA inference completed successfully."
            if cot_success and raw_output
            else (
                "CoT-IQA inference failed or returned empty output: "
                f"{cot_result.get('error') or 'empty model response'}"
            )
        ),
        details={
            "generated_token_count": cot_result.get(
                "generated_token_count"
            ),
            "inference_time_seconds": cot_result.get(
                "inference_time_seconds"
            ),
        },
    )

    if not cot_success or not raw_output:
        return {
            "status": "partial",
            "checks": checks,
            "check_count": len(checks),
            "passed_count": sum(
                check["status"] == "passed"
                for check in checks
            ),
            "warning_count": sum(
                check["status"] == "warning"
                for check in checks
            ),
            "failed_count": sum(
                check["status"] == "failed"
                for check in checks
            ),
            "warnings": warnings,
            "errors": errors,
            "summary": {
                "image_valid": image_valid,
                "pyiqa_metrics": sorted(pyiqa_scores),
                "cot_success": False,
                "cot_complete": False,
                "predicted_mos": None,
                "selected_expert": None,
                "cot_semantic_checks_skipped": True,
            },
        }

    # 4. Truncation
    hit_limit = bool(
        cot_result.get("hit_max_new_tokens", False)
    )

    _record_check(
        checks,
        warnings,
        errors,
        name="generation_truncation",
        status="warning" if hit_limit else "passed",
        message=(
            "Generation reached max_new_tokens and may be truncated."
            if hit_limit
            else "Generation did not reach max_new_tokens."
        ),
        details={
            "ended_with_eos": cot_result.get("ended_with_eos"),
            "generated_token_count": cot_result.get(
                "generated_token_count"
            ),
        },
    )

    # 5. Six-section completeness
    complete = bool(parsed.get("complete", False))
    missing_sections = parsed.get("missing_sections", []) or []

    _record_check(
        checks,
        warnings,
        errors,
        name="cot_structure",
        status="passed" if complete else "warning",
        message=(
            "All six CoT-IQA sections were parsed."
            if complete
            else (
                "CoT-IQA output is incomplete; missing sections: "
                f"{missing_sections}"
            )
        ),
        details={
            "missing_sections": missing_sections,
            "parser_warnings": parsed.get("warnings", []) or [],
        },
    )

    parser_warnings = parsed.get("warnings", []) or []

    _record_check(
        checks,
        warnings,
        errors,
        name="parser_integrity",
        status="warning" if parser_warnings else "passed",
        message=(
            "Parser warnings: "
            + " | ".join(str(item) for item in parser_warnings)
            if parser_warnings
            else "No parser warnings were reported."
        ),
        details={
            "parser_warnings": parser_warnings,
        },
    )

    if not complete or hit_limit:
        partial_quality = (
            parsed.get("quality_prediction")
            or {}
        )

        partial_expert_routing = (
            parsed.get("expert_routing")
            or {}
        )

        return {
            "status": "partial",
            "checks": checks,
            "check_count": len(checks),
            "passed_count": sum(
                check["status"] == "passed"
                for check in checks
            ),
            "warning_count": sum(
                check["status"] == "warning"
                for check in checks
            ),
            "failed_count": sum(
                check["status"] == "failed"
                for check in checks
            ),
            "warnings": warnings,
            "errors": errors,
            "summary": {
                "image_valid": image_valid,
                "pyiqa_metrics": sorted(pyiqa_scores),
                "cot_success": cot_success,
                "cot_complete": complete,
                "hit_max_new_tokens": hit_limit,
                "missing_sections": missing_sections,
                "predicted_mos": partial_quality.get(
                    "predicted_mos"
                ),
                "selected_expert": partial_expert_routing.get(
                    "selected_expert"
                ),
                "cot_semantic_checks_skipped": True,
                "skip_reason": (
                    "generation_truncated"
                    if hit_limit
                    else "incomplete_cot_structure"
                ),
            },
        }

    # 6. Localization bounding boxes
    localization = parsed.get("localization", []) or []
    invalid_regions = [
        item.get("region")
        for item in localization
        if not item.get("bbox_valid", False)
    ]

    _record_check(
        checks,
        warnings,
        errors,
        name="localization_bboxes",
        status="warning" if invalid_regions else "passed",
        message=(
            f"Invalid bounding boxes found in regions: {invalid_regions}"
            if invalid_regions
            else "All parsed localization bounding boxes are valid."
        ),
        details={
            "region_count": len(localization),
            "invalid_regions": invalid_regions,
        },
    )

    # 7. Diagnosis mean integrity
    diagnosis = parsed.get("diagnosis", {}) or {}
    diagnosis_rows = diagnosis.get("rows", []) or {}

    row_scores = [
        float(row["score"])
        for row in diagnosis_rows
        if _finite_number(row.get("score"))
    ]

    parsed_mean = diagnosis.get("mean_score")
    calculated_mean = (
        round(fmean(row_scores), 6)
        if row_scores
        else None
    )

    mean_matches_rows = (
        _finite_number(parsed_mean)
        and calculated_mean is not None
        and math.isclose(
            float(parsed_mean),
            calculated_mean,
            abs_tol=0.01,
        )
    )

    _record_check(
        checks,
        warnings,
        errors,
        name="diagnosis_mean",
        status="passed" if mean_matches_rows else "warning",
        message=(
            "Diagnosis mean matches the parsed dimension scores."
            if mean_matches_rows
            else (
                "Diagnosis mean does not match the parsed "
                "dimension scores."
            )
        ),
        details={
            "parsed_mean": parsed_mean,
            "calculated_mean": calculated_mean,
            "row_count": len(row_scores),
        },
    )

    # 8. Reasoning-text mean consistency
    quality = parsed.get("quality_prediction", {}) or {}
    reasoning = str(quality.get("reasoning", "") or "")
    reasoning_match = _REASONING_MEAN_RE.search(reasoning)

    reasoning_mean = (
        float(reasoning_match.group(1))
        if reasoning_match
        else None
    )

    if reasoning_mean is None:
        _record_check(
            checks,
            warnings,
            errors,
            name="reasoning_mean_consistency",
            status="passed",
            message=(
                "No explicit diagnosis mean was found in the "
                "quality reasoning text."
            ),
        )

    elif calculated_mean is None:
        _record_check(
            checks,
            warnings,
            errors,
            name="reasoning_mean_consistency",
            status="warning",
            message=(
                "The reasoning text contains a diagnosis mean, "
                "but no dimension scores are available for verification."
            ),
            details={
                "reasoning_mean": reasoning_mean,
            },
        )

    else:
        reasoning_matches = math.isclose(
            reasoning_mean,
            calculated_mean,
            abs_tol=0.11,
        )

        _record_check(
            checks,
            warnings,
            errors,
            name="reasoning_mean_consistency",
            status=(
                "passed"
                if reasoning_matches
                else "warning"
            ),
            message=(
                "Reasoning-text diagnosis mean is consistent "
                "with the parsed dimension scores."
                if reasoning_matches
                else (
                    "Reasoning-text diagnosis mean is inconsistent "
                    "with the parsed dimension scores."
                )
            ),
            details={
                "reasoning_mean": reasoning_mean,
                "calculated_mean": calculated_mean,
                "absolute_difference": round(
                    abs(reasoning_mean - calculated_mean),
                    6,
                ),
            },
        )

    # 9. Expert routing
    expert_routing = parsed.get("expert_routing", {}) or {}
    normalization_status = expert_routing.get(
        "normalization_status"
    )

    expert_status = (
        "warning"
        if normalization_status in {"missing", "invalid", None}
        else "passed"
    )

    _record_check(
        checks,
        warnings,
        errors,
        name="expert_routing",
        status=expert_status,
        message=(
            "Expert routing weights are available and valid."
            if expert_status == "passed"
            else "Expert routing weights are missing or invalid."
        ),
        details={
            "weight_sum": expert_routing.get("weight_sum"),
            "normalization_status": normalization_status,
            "normalization_applied": expert_routing.get(
                "normalization_applied"
            ),
            "selected_expert": expert_routing.get(
                "selected_expert"
            ),
            "top_experts": expert_routing.get(
                "top_experts"
            ),
            "selection_ambiguous": expert_routing.get(
                "selection_ambiguous"
            ),
        },
    )

    # 10. MOS validity
    mos = quality.get("predicted_mos")
    mos_valid = (
        _finite_number(mos)
        and bool(quality.get("scale_valid", False))
        and bool(quality.get("in_range", False))
    )

    _record_check(
        checks,
        warnings,
        errors,
        name="quality_prediction",
        status="passed" if mos_valid else "warning",
        message=(
            "Predicted MOS is valid and within the declared scale."
            if mos_valid
            else "Predicted MOS is missing or outside the declared scale."
        ),
        details={
            "predicted_mos": mos,
            "mos_scale": quality.get("mos_scale"),
            "normalized_mos": quality.get("normalized_mos"),
            "scale_valid": quality.get("scale_valid"),
            "in_range": quality.get("in_range"),
        },
    )

    # Overall status
    if not image_valid:
        overall_status = "failed"
    elif not cot_success or not raw_output:
        overall_status = "partial"
    elif errors:
        overall_status = "failed"
    elif warnings:
        overall_status = "warning"
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(
            check["status"] == "passed"
            for check in checks
        ),
        "warning_count": sum(
            check["status"] == "warning"
            for check in checks
        ),
        "failed_count": sum(
            check["status"] == "failed"
            for check in checks
        ),
        "warnings": warnings,
        "errors": errors,
        "summary": {
            "image_valid": image_valid,
            "pyiqa_metrics": sorted(pyiqa_scores),
            "cot_success": cot_success,
            "cot_complete": complete,
            "predicted_mos": mos,
            "selected_expert": expert_routing.get(
                "selected_expert"
            ),
        },
    }
