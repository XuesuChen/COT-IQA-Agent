"""Pairwise image-comparison decision logic for COT-IQA-Agent."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping


def _is_finite_number(value: Any) -> bool:
    """Return whether value is a finite number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _select_winner(
    first_item_id: str,
    second_item_id: str,
    first_score: Any,
    second_score: Any,
    *,
    lower_better: bool,
    tolerance: float = 0.01,
) -> str | None:
    """Select a winner from two numeric scores."""

    if not _is_finite_number(first_score):
        return None

    if not _is_finite_number(second_score):
        return None

    first_value = float(first_score)
    second_value = float(second_score)

    if math.isclose(
        first_value,
        second_value,
        abs_tol=tolerance,
    ):
        return None

    if lower_better:
        return (
            first_item_id
            if first_value < second_value
            else second_item_id
        )

    return (
        first_item_id
        if first_value > second_value
        else second_item_id
    )


def compare_two_image_results(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse PyIQA metrics and structured CoT diagnosis.

    PyIQA metric rankings and CoT diagnosis means are treated as
    primary evidence. Generated MOS is supplementary evidence only.
    """

    items = list(
        state.get("comparison_items", [])
        or []
    )

    if len(items) != 2:
        return {
            "status": "unsupported",
            "winner_item_id": None,
            "loser_item_id": None,
            "confidence": "none",
            "decision_basis": "unsupported_item_count",
            "trusted_vote_counts": {},
            "trusted_vote_total": 0,
            "evidence": {},
            "mos_conflicts_with_primary": False,
            "conflicts": [],
            "warnings": [],
            "errors": [
                "Pairwise comparison requires exactly two images."
            ],
            "rationale": [
                f"Received {len(items)} comparison item(s)."
            ],
        }

    first_item_id = str(items[0]["item_id"])
    second_item_id = str(items[1]["item_id"])

    item_ids = {
        first_item_id,
        second_item_id,
    }

    pyiqa_data = (
        state.get("comparison_pyiqa_results", {})
        or {}
    )

    cot_data = (
        state.get("comparison_cot_results", {})
        or {}
    )

    cot_items = (
        cot_data.get("items", {})
        or {}
    )

    warnings: list[str] = []
    errors: list[str] = []
    conflicts: list[str] = []
    rationale: list[str] = []
    trusted_votes: list[str] = []

    # ---------------------------------------------------------
    # 1. PyIQA evidence
    # ---------------------------------------------------------

    pairwise_metrics = (
        pyiqa_data.get("pairwise", {})
        or {}
    )

    pyiqa_evidence: dict[str, Any] = {}
    metric_winners: list[str] = []

    for metric_name, metric_result in pairwise_metrics.items():
        winner = metric_result.get(
            "winner_item_id"
        )

        if winner in item_ids:
            trusted_votes.append(winner)
            metric_winners.append(winner)

        pyiqa_evidence[metric_name] = {
            "first_item_id": metric_result.get(
                "first_item_id"
            ),
            "second_item_id": metric_result.get(
                "second_item_id"
            ),
            "first_score": metric_result.get(
                "first_score"
            ),
            "second_score": metric_result.get(
                "second_score"
            ),
            "lower_better": metric_result.get(
                "lower_better"
            ),
            "absolute_difference": metric_result.get(
                "absolute_difference"
            ),
            "winner_item_id": winner,
            "result": metric_result.get(
                "result"
            ),
        }

    unique_metric_winners = set(
        metric_winners
    )

    if len(unique_metric_winners) == 1:
        objective_consensus_winner = next(
            iter(unique_metric_winners)
        )

        rationale.append(
            "All available PyIQA metrics agree that "
            f"{objective_consensus_winner} has better quality."
        )

    elif len(unique_metric_winners) > 1:
        objective_consensus_winner = None

        conflicts.append(
            "Available PyIQA metrics disagree on the better image."
        )

    else:
        objective_consensus_winner = None

        warnings.append(
            "No usable pairwise PyIQA evidence is available."
        )

    # ---------------------------------------------------------
    # 2. Structured diagnosis evidence
    # Lower diagnosis mean means less severe degradation.
    # ---------------------------------------------------------

    diagnosis_scores: dict[str, float | None] = {}

    for item_id in item_ids:
        item_result = (
            cot_items.get(item_id, {})
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

        mean_score = diagnosis.get(
            "mean_score"
        )

        diagnosis_scores[item_id] = (
            float(mean_score)
            if _is_finite_number(mean_score)
            else None
        )

    diagnosis_winner = _select_winner(
        first_item_id,
        second_item_id,
        diagnosis_scores[first_item_id],
        diagnosis_scores[second_item_id],
        lower_better=True,
    )

    if diagnosis_winner is not None:
        trusted_votes.append(
            diagnosis_winner
        )

        rationale.append(
            "Structured CoT diagnosis favors "
            f"{diagnosis_winner} because its mean "
            "distortion score is lower."
        )

    else:
        warnings.append(
            "Structured diagnosis could not determine a unique winner."
        )

    diagnosis_evidence = {
        "first_item_id": first_item_id,
        "second_item_id": second_item_id,
        "first_mean_score": diagnosis_scores[
            first_item_id
        ],
        "second_mean_score": diagnosis_scores[
            second_item_id
        ],
        "lower_better": True,
        "winner_item_id": diagnosis_winner,
    }

    # ---------------------------------------------------------
    # 3. Generated MOS — supplementary evidence only
    # ---------------------------------------------------------

    mos_scores: dict[str, float | None] = {}

    for item_id in item_ids:
        item_result = (
            cot_items.get(item_id, {})
            or {}
        )

        parsed = (
            item_result.get("parsed_output")
            or {}
        )

        quality = (
            parsed.get("quality_prediction")
            or {}
        )

        mos = quality.get(
            "predicted_mos"
        )

        mos_scores[item_id] = (
            float(mos)
            if _is_finite_number(mos)
            else None
        )

    mos_winner = _select_winner(
        first_item_id,
        second_item_id,
        mos_scores[first_item_id],
        mos_scores[second_item_id],
        lower_better=False,
    )

    mos_evidence = {
        "first_item_id": first_item_id,
        "second_item_id": second_item_id,
        "first_mos": mos_scores[first_item_id],
        "second_mos": mos_scores[second_item_id],
        "higher_better": True,
        "winner_item_id": mos_winner,
        "used_for_primary_decision": False,
    }

    if mos_winner is None:
        warnings.append(
            "Generated MOS could not determine a unique winner."
        )

    # ---------------------------------------------------------
    # 4. CoT runtime integrity
    # ---------------------------------------------------------

    cot_runtime: dict[str, Any] = {}

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

        cot_runtime[item_id] = {
            "success": success,
            "complete": complete,
            "truncated": truncated,
            "parser_warnings": (
                parsed.get("warnings")
                or []
            ),
            "error": item_result.get("error"),
        }

        if not success:
            errors.append(
                f"CoT-IQA failed for {item_id}: "
                f"{item_result.get('error') or 'unknown error'}"
            )

        elif not complete or truncated:
            warnings.append(
                f"CoT-IQA output for {item_id} "
                "is incomplete or truncated."
            )

    # ---------------------------------------------------------
    # 5. Primary-vote decision
    # ---------------------------------------------------------

    vote_counts = Counter(
        trusted_votes
    )

    first_votes = vote_counts.get(
        first_item_id,
        0,
    )

    second_votes = vote_counts.get(
        second_item_id,
        0,
    )

    total_votes = (
        first_votes + second_votes
    )

    if total_votes == 0:
        winner_item_id = None

    elif first_votes == second_votes:
        winner_item_id = None

        conflicts.append(
            "Primary evidence is tied between the two images."
        )

    else:
        winner_item_id = (
            first_item_id
            if first_votes > second_votes
            else second_item_id
        )

    if winner_item_id == first_item_id:
        loser_item_id = second_item_id

    elif winner_item_id == second_item_id:
        loser_item_id = first_item_id

    else:
        loser_item_id = None

    if (
        objective_consensus_winner is not None
        and diagnosis_winner is not None
        and objective_consensus_winner
        != diagnosis_winner
    ):
        conflicts.append(
            "PyIQA consensus conflicts with structured "
            "CoT diagnosis."
        )

    mos_conflicts_with_primary = (
        winner_item_id is not None
        and mos_winner is not None
        and winner_item_id != mos_winner
    )

    if mos_conflicts_with_primary:
        conflicts.append(
            "Generated MOS conflicts with the primary "
            "PyIQA and diagnosis decision."
        )

        rationale.append(
            "Generated MOS is retained as supplementary evidence "
            "and does not override the primary decision."
        )

    if winner_item_id is not None:
        rationale.append(
            f"{winner_item_id} receives "
            f"{vote_counts[winner_item_id]} of "
            f"{total_votes} primary evidence vote(s)."
        )

    # ---------------------------------------------------------
    # 6. Confidence and status
    # ---------------------------------------------------------

    if winner_item_id is None:
        confidence = "low"

    else:
        support_ratio = (
            vote_counts[winner_item_id]
            / total_votes
        )

        if (
            total_votes >= 3
            and math.isclose(
                support_ratio,
                1.0,
                abs_tol=1e-12,
            )
        ):
            confidence = "high"

        elif support_ratio >= (2.0 / 3.0):
            confidence = "medium"

        else:
            confidence = "low"

        if (
            mos_conflicts_with_primary
            and confidence == "high"
        ):
            confidence = "medium"

    if winner_item_id is None:
        status = "inconclusive"

    elif errors:
        status = "partial"

    elif warnings or conflicts:
        status = "warning"

    else:
        status = "ok"

    return {
        "status": status,
        "winner_item_id": winner_item_id,
        "loser_item_id": loser_item_id,
        "confidence": confidence,
        "decision_basis": (
            "pyiqa_metrics_and_structured_diagnosis"
        ),
        "trusted_vote_counts": {
            first_item_id: first_votes,
            second_item_id: second_votes,
        },
        "trusted_vote_total": total_votes,
        "evidence": {
            "pyiqa": pyiqa_evidence,
            "objective_consensus_winner": (
                objective_consensus_winner
            ),
            "diagnosis": diagnosis_evidence,
            "generated_mos": mos_evidence,
            "cot_runtime": cot_runtime,
        },
        "mos_conflicts_with_primary": (
            mos_conflicts_with_primary
        ),
        "conflicts": conflicts,
        "warnings": warnings,
        "errors": errors,
        "rationale": rationale,
    }
