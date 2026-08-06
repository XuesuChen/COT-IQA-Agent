"""Parser for six-step CoT-IQA model outputs."""

from __future__ import annotations

import re
from typing import Any


SECTION_KEYS = {
    1: "localization",
    2: "attribution",
    3: "diagnosis",
    4: "restoration_suggestion",
    5: "expert_routing",
    6: "quality_prediction",
}

REQUIRED_SECTIONS = tuple(SECTION_KEYS.values())

SECTION_PATTERN = re.compile(
    r"^###\s*Step\s*(?P<number>[1-6])\s*:\s*(?P<title>.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _normalize_label(value: str) -> str:
    """Convert a free-text label to a lowercase snake-case identifier."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


def _parse_number(value: str) -> float | int:
    """Parse a numeric string and preserve integers when possible."""
    number = float(value)

    if number.is_integer():
        return int(number)

    return number


def _parse_sections(raw_output: str) -> tuple[
    dict[str, str],
    dict[str, str],
]:
    """Split a model output into six canonical sections."""

    matches = list(SECTION_PATTERN.finditer(raw_output))
    sections: dict[str, str] = {}
    titles: dict[str, str] = {}

    for index, match in enumerate(matches):
        step_number = int(match.group("number"))
        section_key = SECTION_KEYS[step_number]

        content_start = match.end()
        content_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(raw_output)
        )

        sections[section_key] = raw_output[
            content_start:content_end
        ].strip()

        titles[section_key] = match.group("title").strip()

    return sections, titles


def _parse_localization(section: str) -> list[dict[str, Any]]:
    """Parse degradation regions from Step 1."""

    pattern = re.compile(
        r"^\s*-\s*Region\s+(?P<region>[^:]+?)\s*:\s*"
        r"\[(?P<bbox>[^\]]+)\]\s*--\s*"
        r"(?P<distortion>.+?),\s*"
        r"(?P<severity>Mild|Moderate|Severe)"
        r"(?:\s*\((?P<scope>[^)]+)\))?\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    regions: list[dict[str, Any]] = []

    for match in pattern.finditer(section):
        bbox_values = NUMBER_PATTERN.findall(match.group("bbox"))
        bbox = [_parse_number(value) for value in bbox_values]

        bbox_valid = True
        bbox_error: str | None = None

        if len(bbox) != 4:
            bbox_valid = False
            bbox_error = (
                "Bounding box must contain exactly four coordinates."
            )

        elif bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bbox_valid = False
            bbox_error = (
                "Bounding box must satisfy x2 > x1 and y2 > y1."
            )

        distortion_raw = match.group("distortion").strip()
        severity_raw = match.group("severity").strip()
        scope_match = match.group("scope")

        if scope_match:
            scope_raw = scope_match.strip()
            scope = scope_raw.lower()
            scope_inferred = False
        else:
            # Phase-2 local-region outputs omit the explicit "(local)" marker.
            scope_raw = None
            scope = "local"
            scope_inferred = True

        regions.append(
            {
                "region": match.group("region").strip(),
                "bbox": bbox,
                "bbox_valid": bbox_valid,
                "bbox_error": bbox_error,
                "distortion": _normalize_label(distortion_raw),
                "distortion_raw": distortion_raw,
                "severity": severity_raw.lower(),
                "scope": scope,
                "scope_raw": scope_raw,
                "scope_inferred": scope_inferred,
            }
        )

    return regions

def _parse_attribution(section: str) -> dict[str, Any]:
    """Extract structured signals from Step 2."""

    count_match = re.search(
        r"contains\s+(?P<count>\d+)\s+degradation type",
        section,
        flags=re.IGNORECASE,
    )

    primary_match = re.search(
        r"primary impairment is\s+(?P<primary>.+?)"
        r"(?:\.|\n|$)",
        section,
        flags=re.IGNORECASE,
    )

    degradation_count = (
        int(count_match.group("count"))
        if count_match
        else None
    )

    primary_impairment_raw = (
        primary_match.group("primary").strip()
        if primary_match
        else None
    )

    return {
        "text": section.strip(),
        "degradation_count": degradation_count,
        "primary_impairment": (
            _normalize_label(primary_impairment_raw)
            if primary_impairment_raw
            else None
        ),
        "primary_impairment_raw": primary_impairment_raw,
    }


def _parse_diagnosis(section: str) -> dict[str, Any]:
    """Parse the Markdown diagnosis table from Step 3."""

    rows: list[dict[str, Any]] = []
    dimensions: dict[str, dict[str, Any]] = {}

    for line in section.splitlines():
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        cells = [
            cell.strip()
            for cell in stripped.strip("|").split("|")
        ]

        if len(cells) < 3:
            continue

        dimension, score_text, description = cells[:3]

        if dimension.casefold() == "dimension":
            continue

        if set(dimension) <= {"-", ":"}:
            continue

        score_match = NUMBER_PATTERN.search(score_text)

        if score_match is None:
            continue

        score = float(score_match.group())

        row = {
            "dimension": _normalize_label(dimension),
            "dimension_raw": dimension,
            "score": score,
            "description": description,
        }

        rows.append(row)
        dimensions[row["dimension"]] = {
            "score": score,
            "description": description,
        }

    mean_score = (
        sum(row["score"] for row in rows) / len(rows)
        if rows
        else None
    )

    return {
        "rows": rows,
        "dimensions": dimensions,
        "mean_score": (
            round(mean_score, 6)
            if mean_score is not None
            else None
        ),
    }


def _parse_restoration(section: str) -> list[dict[str, Any]]:
    """Parse regional restoration suggestions from Step 4."""

    pattern = re.compile(
        r"^-\s*Region\s+(?P<region>[^(:]+?)\s*"
        r"\((?P<distortion>[^)]+)\)\s*:\s*"
        r"(?P<action>.+?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    suggestions: list[dict[str, Any]] = []

    for match in pattern.finditer(section):
        action = match.group("action").strip().rstrip(".")
        distortion_raw = match.group("distortion").strip()

        suggestions.append(
            {
                "region": match.group("region").strip(),
                "distortion": _normalize_label(distortion_raw),
                "distortion_raw": distortion_raw,
                "action": action,
            }
        )

    return suggestions


def _parse_expert_routing(section: str) -> dict[str, Any]:
    """Parse raw and normalized expert-routing weights from Step 5."""

    pattern = re.compile(
        r"^\s*-\s*(?P<expert>[^:\n]+?Expert)\s*:\s*"
        r"(?P<weight>[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    experts: dict[str, float] = {}
    experts_raw: dict[str, float] = {}

    for match in pattern.finditer(section):
        expert_raw = match.group("expert").strip()
        weight = float(match.group("weight"))

        expert_key = _normalize_label(
            re.sub(
                r"\s+Expert$",
                "",
                expert_raw,
                flags=re.IGNORECASE,
            )
        )

        experts[expert_key] = weight
        experts_raw[expert_raw] = weight

    weight_sum = sum(experts.values()) if experts else None
    contains_negative_weight = any(
        weight < 0
        for weight in experts.values()
    )

    normalized_weights: dict[str, float] = {}

    if (
        weight_sum is not None
        and weight_sum > 0
        and not contains_negative_weight
    ):
        normalized_weights = {
            expert: round(weight / weight_sum, 6)
            for expert, weight in experts.items()
        }

    ranking_weights = normalized_weights or experts

    ranked_experts = sorted(
        ranking_weights.items(),
        key=lambda item: (-item[1], item[0]),
    )

    top_experts: list[str] = []
    top_weight_gap: float | None = None

    if ranked_experts:
        top_weight = ranked_experts[0][1]

        top_experts = [
            expert
            for expert, weight in ranked_experts
            if abs(weight - top_weight) <= 1e-6
        ]

        if len(ranked_experts) >= 2:
            top_weight_gap = round(
                ranked_experts[0][1] - ranked_experts[1][1],
                6,
            )

    selected_expert = (
        top_experts[0]
        if len(top_experts) == 1
        else None
    )

    if not experts:
        normalization_status = "missing"
    elif contains_negative_weight or weight_sum is None or weight_sum <= 0:
        normalization_status = "invalid"
    elif abs(weight_sum - 1.0) <= 1e-6:
        normalization_status = "exact"
    else:
        normalization_status = "normalized"

    return {
        "weights": experts,
        "weights_raw": experts_raw,
        "normalized_weights": normalized_weights,
        "weight_sum": (
            round(weight_sum, 6)
            if weight_sum is not None
            else None
        ),
        "selected_expert": selected_expert,
        "top_experts": top_experts,
        "selection_ambiguous": len(top_experts) > 1,
        "top_weight_gap": top_weight_gap,
        "normalization_status": normalization_status,
        "normalization_applied": normalization_status == "normalized",
    }

def _parse_quality_prediction(section: str) -> dict[str, Any]:
    """Parse predicted MOS and reasoning from Step 6."""

    mos_match = re.search(
        r"Predicted\s+MOS\s*:\s*"
        r"(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*/\s*"
        r"(?P<scale>[-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        section,
        flags=re.IGNORECASE,
    )

    reasoning_match = re.search(
        r"Reasoning\s*:\s*(?P<reasoning>.+)$",
        section,
        flags=re.IGNORECASE | re.DOTALL,
    )

    predicted_mos = (
        float(mos_match.group("score"))
        if mos_match
        else None
    )

    mos_scale = (
        float(mos_match.group("scale"))
        if mos_match
        else None
    )

    reasoning = (
        reasoning_match.group("reasoning").strip()
        if reasoning_match
        else None
    )

    scale_valid = mos_scale is not None and mos_scale > 0

    in_range = (
        predicted_mos is not None
        and scale_valid
        and 0 <= predicted_mos <= mos_scale
    )

    normalized_mos = (
        predicted_mos / mos_scale
        if in_range
        else None
    )

    return {
        "predicted_mos": predicted_mos,
        "mos_scale": mos_scale,
        "scale_valid": scale_valid,
        "in_range": in_range,
        "normalized_mos": (
            round(normalized_mos, 6)
            if normalized_mos is not None
            else None
        ),
        "reasoning": reasoning,
    }

def parse_cot_output(
    raw_output: str,
    keep_raw_output: bool = True,
) -> dict[str, Any]:
    """Parse one six-step CoT-IQA model response.

    The parser does not treat the generated MOS as a calibrated final IQA
    score. It only extracts the value reported by the generative model.
    """

    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string.")

    raw_output = raw_output.strip()

    if not raw_output:
        raise ValueError("raw_output must not be empty.")

    sections, section_titles = _parse_sections(raw_output)

    missing_sections = [
        section_name
        for section_name in REQUIRED_SECTIONS
        if not sections.get(section_name)
    ]

    localization = _parse_localization(
        sections.get("localization", "")
    )
    attribution = _parse_attribution(
        sections.get("attribution", "")
    )
    diagnosis = _parse_diagnosis(
        sections.get("diagnosis", "")
    )
    restoration = _parse_restoration(
        sections.get("restoration_suggestion", "")
    )
    expert_routing = _parse_expert_routing(
        sections.get("expert_routing", "")
    )
    quality_prediction = _parse_quality_prediction(
        sections.get("quality_prediction", "")
    )

    warnings: list[str] = []

    if sections.get("localization") and not localization:
        warnings.append(
            "Localization section was found, but no regions were parsed."
        )

    for region in localization:
        if not region["bbox_valid"]:
            warnings.append(
                f"Localization region {region['region']} has an invalid "
                f"bounding box: {region['bbox_error']}"
            )

    if (
        sections.get("diagnosis")
        and not diagnosis["rows"]
    ):
        warnings.append(
            "Diagnosis section was found, but no table rows were parsed."
        )

    if (
        sections.get("restoration_suggestion")
        and not restoration
    ):
        warnings.append(
            "Restoration section was found, but no suggestions were parsed."
        )

    if (
        sections.get("expert_routing")
        and not expert_routing["weights"]
    ):
        warnings.append(
            "Expert-routing section was found, but no weights were parsed."
        )

    if (
        sections.get("quality_prediction")
        and quality_prediction["predicted_mos"] is None
    ):
        warnings.append(
            "Quality-prediction section was found, but MOS was not parsed."
        )

    elif sections.get("quality_prediction"):
        if not quality_prediction["scale_valid"]:
            warnings.append(
                "MOS scale must be greater than zero."
            )

        elif not quality_prediction["in_range"]:
            warnings.append(
                "Predicted MOS is outside the declared MOS range."
            )

    if expert_routing["normalization_status"] == "invalid":
        warnings.append(
            "Expert-routing weights are invalid and could not be normalized."
        )

    elif expert_routing["weight_sum"] is not None:
        if abs(expert_routing["weight_sum"] - 1.0) > 0.10:
            warnings.append(
                "Expert-routing weights materially deviate from a sum of 1."
            )

    return {
        "complete": not missing_sections,
        "missing_sections": missing_sections,
        "warnings": warnings,
        "section_titles": section_titles,
        "sections": sections,
        "localization": localization,
        "attribution": attribution,
        "diagnosis": diagnosis,
        "restoration_suggestion": restoration,
        "expert_routing": expert_routing,
        "quality_prediction": quality_prediction,
        "raw_output": raw_output if keep_raw_output else None,
    }
