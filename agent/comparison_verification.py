"""Verification logic for pairwise image comparison."""

from __future__ import annotations

from typing import Any, Mapping


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

    check = {
        "name": name,
        "status": status,
        "message": message,
        "details": details or {},
    }

    checks.append(check)

    if status == "warning":
        warnings.append(message)

    elif status == "failed":
        errors.append(message)


def _build_result(
    *,
    status: str,
    checks: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Build a consistent verification result."""

    return {
        "status": status,
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
        "summary": summary,
    }


def verify_comparison_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify pairwise comparison inputs, evidence and decision."""

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    items = list(
        state.get("comparison_items", [])
        or []
    )

    item_ids = [
        str(item.get("item_id"))
        for item in items
        if item.get("item_id")
    ]

    if len(items) != 2:
        message = (
            "Pairwise comparison verification requires "
            f"exactly two images; received {len(items)}."
        )

        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_input_count",
            status="failed",
            message=message,
            details={
                "item_count": len(items),
                "item_ids": item_ids,
            },
        )

        return _build_result(
            status="failed",
            checks=checks,
            warnings=warnings,
            errors=errors,
            summary={
                "item_count": len(items),
                "item_ids": item_ids,
                "images_valid": False,
                "winner_item_id": None,
                "loser_item_id": None,
                "confidence": "none",
                "verification_short_circuited": True,
                "skip_reason": "invalid_item_count",
            },
        )

    _record_check(
        checks,
        warnings,
        errors,
        name="comparison_input_count",
        status="passed",
        message="Exactly two comparison images are available.",
        details={
            "item_count": len(items),
            "item_ids": item_ids,
        },
    )

    # ---------------------------------------------------------
    # Image validation
    # ---------------------------------------------------------

    image_analysis = (
        state.get(
            "comparison_image_analysis",
            {},
        )
        or {}
    )

    all_valid = bool(
        image_analysis.get("all_valid", False)
    )

    valid_count = int(
        image_analysis.get("valid_count", 0)
        or 0
    )

    duplicate_paths = list(
        image_analysis.get("duplicate_paths", [])
        or []
    )

    if all_valid and valid_count == 2 and not duplicate_paths:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_image_validation",
            status="passed",
            message=(
                "Both comparison images passed validation "
                "and use distinct paths."
            ),
            details={
                "valid_count": valid_count,
                "duplicate_paths": duplicate_paths,
            },
        )

    else:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_image_validation",
            status="failed",
            message=(
                "Comparison image validation failed or "
                "duplicate image paths were detected."
            ),
            details={
                "valid_count": valid_count,
                "all_valid": all_valid,
                "duplicate_paths": duplicate_paths,
            },
        )

        return _build_result(
            status="failed",
            checks=checks,
            warnings=warnings,
            errors=errors,
            summary={
                "item_count": 2,
                "item_ids": item_ids,
                "images_valid": False,
                "winner_item_id": None,
                "loser_item_id": None,
                "confidence": "none",
                "verification_short_circuited": True,
                "skip_reason": "invalid_comparison_images",
            },
        )

    # ---------------------------------------------------------
    # PyIQA comparison integrity
    # ---------------------------------------------------------

    pyiqa = (
        state.get(
            "comparison_pyiqa_results",
            {},
        )
        or {}
    )

    pyiqa_skipped = bool(
        pyiqa.get("skipped", False)
    )

    evaluations = (
        pyiqa.get("evaluations", {})
        or {}
    )

    rankings = (
        pyiqa.get("rankings", {})
        or {}
    )

    pairwise = (
        pyiqa.get("pairwise", {})
        or {}
    )

    metric_errors: list[str] = []

    for item_id, evaluation in evaluations.items():
        for error in evaluation.get("errors", []) or []:
            metric_errors.append(
                f"{item_id}: {error}"
            )

    if pyiqa_skipped or not pairwise:
        _record_check(
            checks,
            warnings,
            errors,
            name="pyiqa_comparison",
            status="warning",
            message=(
                "No usable pairwise PyIQA evidence is available."
            ),
            details={
                "skipped": pyiqa_skipped,
                "evaluation_count": len(evaluations),
                "ranking_metrics": sorted(rankings),
                "pairwise_metrics": sorted(pairwise),
            },
        )

    elif metric_errors:
        _record_check(
            checks,
            warnings,
            errors,
            name="pyiqa_comparison",
            status="warning",
            message=(
                "PyIQA comparison completed with "
                f"{len(metric_errors)} metric error(s)."
            ),
            details={
                "metric_errors": metric_errors,
                "pairwise_metrics": sorted(pairwise),
            },
        )

    else:
        _record_check(
            checks,
            warnings,
            errors,
            name="pyiqa_comparison",
            status="passed",
            message=(
                "Pairwise PyIQA evidence is available "
                "without metric errors."
            ),
            details={
                "pairwise_metrics": sorted(pairwise),
                "evaluation_count": len(evaluations),
            },
        )

    # ---------------------------------------------------------
    # CoT-IQA comparison integrity
    # ---------------------------------------------------------

    cot_data = (
        state.get(
            "comparison_cot_results",
            {},
        )
        or {}
    )

    cot_items = (
        cot_data.get("items", {})
        or {}
    )

    success_count = int(
        cot_data.get("success_count", 0)
        or 0
    )

    failed_count = int(
        cot_data.get("failed_count", 0)
        or 0
    )

    cot_runtime_details: dict[str, Any] = {}

    incomplete_items: list[str] = []
    truncated_items: list[str] = []
    parser_warning_items: list[str] = []
    failed_items: list[str] = []

    for item_id in item_ids:
        item_result = (
            cot_items.get(item_id, {})
            or {}
        )

        parsed = (
            item_result.get("parsed_output")
            or {}
        )

        success = bool(
            item_result.get("success", False)
        )

        complete = bool(
            parsed.get("complete", False)
        )

        truncated = bool(
            item_result.get(
                "hit_max_new_tokens",
                False,
            )
        )

        parser_warnings = list(
            parsed.get("warnings", [])
            or []
        )

        cot_runtime_details[item_id] = {
            "success": success,
            "complete": complete,
            "truncated": truncated,
            "parser_warnings": parser_warnings,
            "error": item_result.get("error"),
        }

        if not success:
            failed_items.append(item_id)

        if success and not complete:
            incomplete_items.append(item_id)

        if truncated:
            truncated_items.append(item_id)

        if parser_warnings:
            parser_warning_items.append(item_id)

    if success_count == 2 and not (
        incomplete_items
        or truncated_items
        or parser_warning_items
    ):
        _record_check(
            checks,
            warnings,
            errors,
            name="cot_iqa_comparison",
            status="passed",
            message=(
                "CoT-IQA completed successfully for both images."
            ),
            details=cot_runtime_details,
        )

    elif success_count == 0:
        _record_check(
            checks,
            warnings,
            errors,
            name="cot_iqa_comparison",
            status="failed",
            message=(
                "CoT-IQA failed for both comparison images."
            ),
            details={
                "success_count": success_count,
                "failed_count": failed_count,
                "failed_items": failed_items,
                "runtime": cot_runtime_details,
            },
        )

    else:
        _record_check(
            checks,
            warnings,
            errors,
            name="cot_iqa_comparison",
            status="warning",
            message=(
                "CoT-IQA comparison completed partially or "
                "contains incomplete parser output."
            ),
            details={
                "success_count": success_count,
                "failed_count": failed_count,
                "failed_items": failed_items,
                "incomplete_items": incomplete_items,
                "truncated_items": truncated_items,
                "parser_warning_items": parser_warning_items,
                "runtime": cot_runtime_details,
            },
        )

    # ---------------------------------------------------------
    # Final comparison decision
    # ---------------------------------------------------------

    decision = (
        state.get("comparison_result", {})
        or {}
    )

    decision_status = str(
        decision.get("status", "")
    )

    winner_item_id = decision.get(
        "winner_item_id"
    )

    loser_item_id = decision.get(
        "loser_item_id"
    )

    confidence = decision.get(
        "confidence",
        "none",
    )

    decision_errors = list(
        decision.get("errors", [])
        or []
    )

    decision_warnings = list(
        decision.get("warnings", [])
        or []
    )

    decision_conflicts = list(
        decision.get("conflicts", [])
        or []
    )

    winner_valid = (
        winner_item_id in item_ids
    )

    loser_valid = (
        loser_item_id in item_ids
        and loser_item_id != winner_item_id
    )

    if decision_errors:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_decision",
            status="failed",
            message=(
                "Comparison decision contains "
                f"{len(decision_errors)} error(s)."
            ),
            details={
                "status": decision_status,
                "errors": decision_errors,
            },
        )

    elif winner_item_id is None:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_decision",
            status="warning",
            message=(
                "Comparison evidence did not produce "
                "a unique winner."
            ),
            details={
                "status": decision_status,
                "confidence": confidence,
            },
        )

    elif not winner_valid or not loser_valid:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_decision",
            status="failed",
            message=(
                "Comparison winner or loser identifier "
                "is inconsistent with the input items."
            ),
            details={
                "item_ids": item_ids,
                "winner_item_id": winner_item_id,
                "loser_item_id": loser_item_id,
            },
        )

    else:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_decision",
            status="passed",
            message=(
                f"Comparison selected {winner_item_id} "
                "as the better-quality image."
            ),
            details={
                "status": decision_status,
                "winner_item_id": winner_item_id,
                "loser_item_id": loser_item_id,
                "confidence": confidence,
                "decision_basis": decision.get(
                    "decision_basis"
                ),
            },
        )

    # ---------------------------------------------------------
    # Vote consistency
    # ---------------------------------------------------------

    vote_counts = (
        decision.get(
            "trusted_vote_counts",
            {},
        )
        or {}
    )

    declared_vote_total = int(
        decision.get(
            "trusted_vote_total",
            0,
        )
        or 0
    )

    calculated_vote_total = sum(
        int(value or 0)
        for value in vote_counts.values()
        if isinstance(value, int)
    )

    vote_keys_valid = set(
        vote_counts
    ).issubset(set(item_ids))

    winner_has_max_votes = True

    if winner_item_id is not None and vote_counts:
        winner_votes = int(
            vote_counts.get(
                winner_item_id,
                0,
            )
            or 0
        )

        winner_has_max_votes = all(
            winner_votes >= int(value or 0)
            for value in vote_counts.values()
            if isinstance(value, int)
        )

    if (
        declared_vote_total != calculated_vote_total
        or not vote_keys_valid
        or not winner_has_max_votes
    ):
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_vote_integrity",
            status="failed",
            message=(
                "Comparison vote totals or winner selection "
                "are internally inconsistent."
            ),
            details={
                "vote_counts": vote_counts,
                "declared_vote_total": declared_vote_total,
                "calculated_vote_total": calculated_vote_total,
                "vote_keys_valid": vote_keys_valid,
                "winner_has_max_votes": winner_has_max_votes,
            },
        )

    elif declared_vote_total == 0:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_vote_integrity",
            status="warning",
            message=(
                "No trusted primary evidence votes "
                "were available."
            ),
            details={
                "vote_counts": vote_counts,
                "trusted_vote_total": declared_vote_total,
            },
        )

    else:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_vote_integrity",
            status="passed",
            message=(
                "Trusted evidence vote totals are consistent."
            ),
            details={
                "vote_counts": vote_counts,
                "trusted_vote_total": declared_vote_total,
            },
        )

    # ---------------------------------------------------------
    # Conflict transparency
    # ---------------------------------------------------------

    if decision_conflicts:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_conflicts",
            status="warning",
            message=(
                "Comparison contains "
                f"{len(decision_conflicts)} evidence conflict(s)."
            ),
            details={
                "conflicts": decision_conflicts,
                "mos_conflicts_with_primary": decision.get(
                    "mos_conflicts_with_primary",
                    False,
                ),
            },
        )

    elif decision_warnings:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_conflicts",
            status="warning",
            message=(
                "Comparison decision contains "
                f"{len(decision_warnings)} warning(s)."
            ),
            details={
                "warnings": decision_warnings,
            },
        )

    else:
        _record_check(
            checks,
            warnings,
            errors,
            name="comparison_conflicts",
            status="passed",
            message=(
                "No unresolved comparison evidence "
                "conflicts were reported."
            ),
            details={},
        )

    failed_checks = sum(
        check["status"] == "failed"
        for check in checks
    )

    warning_checks = sum(
        check["status"] == "warning"
        for check in checks
    )

    if failed_checks:
        overall_status = "partial"

    elif winner_item_id is None:
        overall_status = "inconclusive"

    elif warning_checks:
        overall_status = "warning"

    else:
        overall_status = "ok"

    return _build_result(
        status=overall_status,
        checks=checks,
        warnings=warnings,
        errors=errors,
        summary={
            "item_count": 2,
            "item_ids": item_ids,
            "images_valid": True,
            "pyiqa_pairwise_metrics": sorted(pairwise),
            "cot_success_count": success_count,
            "cot_failed_count": failed_count,
            "winner_item_id": winner_item_id,
            "loser_item_id": loser_item_id,
            "confidence": confidence,
            "decision_status": decision_status,
            "trusted_vote_total": declared_vote_total,
            "conflict_count": len(decision_conflicts),
            "mos_conflicts_with_primary": bool(
                decision.get(
                    "mos_conflicts_with_primary",
                    False,
                )
            ),
        },
    )
