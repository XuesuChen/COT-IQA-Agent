"""Gradio user interface for COT-IQA-Agent."""

from __future__ import annotations

import traceback

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.graph import app as agent_app
from agent.state import create_initial_state


DEFAULT_SINGLE_QUERY = (
    "请分析这张图片的整体质量，识别主要失真、严重程度和影响区域，"
    "给出修复建议，并结合相关 IQA 论文知识进行解释。"
)

DEFAULT_COMPARISON_QUERY = (
    "请比较这两张图片的整体质量，识别各自的主要失真，"
    "判断哪张图片质量更好，并结合相关 IQA 论文知识解释判断依据。"
)


def _json_safe(value: Any) -> Any:
    """Convert nested workflow output into JSON-compatible data."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep useful workflow fields while avoiding excessive UI output."""

    return _json_safe(
        {
            "request_id": result.get("request_id"),
            "route": result.get("route"),
            "execution_trace": result.get(
                "execution_trace",
                [],
            ),
            "image_metadata": result.get(
                "image_metadata",
                {},
            ),
            "image_analysis": result.get(
                "image_analysis",
                {},
            ),
            "pyiqa_results": result.get(
                "pyiqa_results",
                {},
            ),
            "cot_iqa_result": result.get(
                "cot_iqa_result",
                {},
            ),
            "comparison_items": result.get(
                "comparison_items",
                [],
            ),
            "comparison_image_metadata": result.get(
                "comparison_image_metadata",
                {},
            ),
            "comparison_pyiqa_results": result.get(
                "comparison_pyiqa_results",
                {},
            ),
            "comparison_cot_results": result.get(
                "comparison_cot_results",
                {},
            ),
            "comparison_result": result.get(
                "comparison_result",
                {},
            ),
            "rag_context": result.get(
                "rag_context",
                [],
            ),
            "verification_result": result.get(
                "verification_result",
                {},
            ),
            "final_result": result.get(
                "final_result",
                {},
            ),
            "final_report": result.get(
                "final_report",
                "",
            ),
            "errors": result.get(
                "errors",
                [],
            ),
        }
    )


def _format_rag_sources(
    rag_context: list[dict[str, Any]],
) -> list[str]:
    """Format retrieved paper passages as Markdown."""

    lines: list[str] = []

    for item in rag_context:
        metadata = item.get("metadata", {})

        source = metadata.get(
            "source_path",
            "Unknown source",
        )

        page = metadata.get(
            "page_number",
            "?",
        )

        score = item.get("score")

        if isinstance(score, (int, float)):
            score_text = f"{score:.4f}"
        else:
            score_text = "N/A"

        lines.append(
            f"- **{source}**, page {page}, "
            f"similarity `{score_text}`"
        )

    return lines



_FIELD_LABELS = {
    "region": "区域",
    "bbox": "定位范围",
    "scope": "影响范围",
    "distortion": "失真类型",
    "severity": "严重程度",
    "reason": "判断依据",
    "cause": "可能原因",
    "impact": "质量影响",
    "dimension": "质量维度",
    "score": "评分",
    "mean_score": "平均失真分数",
    "diagnosis_mean": "平均失真分数",
    "calculated_mean": "平均失真分数",
    "action": "建议操作",
    "selected_expert": "选中的专家",
    "top_experts": "优先专家",
    "selection_ambiguous": "路由是否存在歧义",
    "predicted_mos": "预测 MOS",
    "mos_scale": "MOS 量表",
    "normalized_mos": "归一化 MOS",
    "confidence": "置信度",
    "complete": "结构是否完整",
}

_DISTORTION_LABELS = {
    "jpeg_compression": "JPEG 压缩伪影",
    "localized jpeg compression": "局部 JPEG 压缩伪影",
    "jpeg2000_compression": "JPEG2000 压缩伪影",
    "gaussian_blur": "高斯模糊",
    "motion_blur": "运动模糊",
    "defocus_blur": "散焦模糊",
    "gaussian_noise": "高斯噪声",
    "salt_pepper_noise": "椒盐噪声",
    "impulse_noise": "脉冲噪声",
    "brightness_change": "亮度异常",
    "contrast_change": "对比度异常",
    "color_shift": "颜色偏移",
    "blocking": "块效应",
    "ringing": "振铃伪影",
    "sharpness": "清晰度问题",
}

_ACTION_LABELS = {
    "apply_deblurring": "执行去模糊处理",
    "apply deblurring": "执行去模糊处理",
    "apply compression_artifact_reduction": "降低压缩伪影",
    "apply_compression_artifact_reduction": "降低压缩伪影",
    "apply denoising": "执行降噪处理",
    "apply_denoising": "执行降噪处理",
    "apply color correction": "执行颜色校正",
    "apply_color_correction": "执行颜色校正",
    "apply contrast enhancement": "增强图像对比度",
    "apply_contrast_enhancement": "增强图像对比度",
    "apply brightness adjustment": "调整图像亮度",
    "apply_brightness_adjustment": "调整图像亮度",
}

_EXPERT_LABELS = {
    "deblur": "去模糊专家",
    "denoise": "降噪专家",
    "decompress": "压缩伪影修复专家",
    "color": "颜色校正专家",
    "artifact": "通用伪影修复专家",
}

_ITEM_LABELS = {
    "image_1": "图片 1",
    "image_2": "图片 2",
}


_DIMENSION_LABELS = {
    "blur": "模糊程度",
    "noise": "噪声程度",
    "compression": "压缩损伤",
    "color": "颜色表现",
    "artifact": "综合伪影",
}

_WARNING_LABELS = {
    (
        "Reasoning-text diagnosis mean is inconsistent "
        "with the parsed dimension scores."
    ): (
        "推理文本中的平均失真分数与结构化维度计算结果"
        "存在轻微不一致。系统采用结构化计算结果作为主要依据。"
    ),
}


_DESCRIPTION_LABELS = {
    "Nearly no visible blur":
        "几乎没有可见模糊。",
    "Significant blur, fine details lost":
        "存在明显模糊，部分细节已经丢失。",
    "Severe block artifacts dominate the image":
        "图像中存在严重的块状压缩伪影。",
    "Minimal composite artifacts":
        "综合伪影较轻，整体影响较小。",
    "Slight edge softening in some regions":
        "部分区域存在轻微的边缘软化。",
    "Nearly noise-free, clean image":
        "图像整体较干净，几乎没有明显噪声。",
    "Slight blocking in smooth gradient regions":
        "平滑渐变区域存在轻微块效应。",
    "Color appears accurate and natural":
        "图像颜色基本准确且自然。",
    "Significant multi-degradation interaction artifacts":
        "多种退化叠加后产生了较明显的综合伪影。",
}


def _display_value(value: Any) -> str:
    """Convert structured values into readable Chinese text."""

    if value is None:
        return "未提供"

    if isinstance(value, bool):
        return "是" if value else "否"

    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")

    if isinstance(value, int):
        return str(value)

    if isinstance(value, str):
        stripped = value.strip()
        normalized = stripped.lower()

        if normalized in _DISTORTION_LABELS:
            return _DISTORTION_LABELS[normalized]

        if normalized in _ACTION_LABELS:
            return _ACTION_LABELS[normalized]

        if normalized in _ITEM_LABELS:
            return _ITEM_LABELS[normalized]

        severity_labels = {
            "mild": "轻度",
            "moderate": "中度",
            "severe": "重度",
            "global": "全局",
            "local": "局部",
            "high": "高",
            "medium": "中",
            "low": "低",
            "ok": "通过",
            "warning": "存在警告",
            "passed": "通过",
            "failed": "失败",
        }

        if normalized in severity_labels:
            return severity_labels[normalized]

        return stripped.replace("_", " ")

    return str(value)



def _display_expert(value: Any) -> str:
    """Translate an expert name without affecting diagnosis dimensions."""

    normalized = str(value).strip().lower()

    return _EXPERT_LABELS.get(
        normalized,
        _display_value(value),
    )


def _display_dimension(value: Any) -> str:
    """Translate an IQA diagnosis dimension."""

    normalized = str(value).strip().lower()

    return _DIMENSION_LABELS.get(
        normalized,
        _display_value(value),
    )


def _translate_warning(value: Any) -> str:
    """Translate verification warnings for UI display."""

    warning_text = str(value).strip()

    return _WARNING_LABELS.get(
        warning_text,
        warning_text,
    )


def _translate_description(value: Any) -> str:
    """Translate common diagnosis descriptions."""

    text_value = str(value).strip()

    return _DESCRIPTION_LABELS.get(
        text_value,
        text_value,
    )


def _field_label(key: str) -> str:
    """Translate common structured field names."""

    return _FIELD_LABELS.get(
        key,
        key.replace("_", " "),
    )


def _render_nested(
    value: Any,
    *,
    level: int = 0,
) -> list[str]:
    """Render arbitrary nested structures as readable Markdown bullets."""

    lines: list[str] = []
    indent = "  " * level

    hidden_fields = {
        "distortion_raw",
        "weights_raw",
        "bbox_error",
        "normalization_applied",
        "scale_valid",
        "in_range",
        "dimension_raw",
        "primary_impairment_raw",
        "weights_raw",
    }

    if isinstance(value, dict):
        for key, item in value.items():
            if key in hidden_fields:
                continue

            label = _field_label(str(key))

            if isinstance(item, (dict, list)):
                if not item:
                    continue

                lines.append(
                    f"{indent}- **{label}**："
                )
                lines.extend(
                    _render_nested(
                        item,
                        level=level + 1,
                    )
                )
            else:
                lines.append(
                    f"{indent}- **{label}**："
                    f"{_display_value(item)}"
                )

    elif isinstance(value, list):
        for index, item in enumerate(value, start=1):
            if isinstance(item, (dict, list)):
                lines.append(
                    f"{indent}- **第 {index} 项**"
                )
                lines.extend(
                    _render_nested(
                        item,
                        level=level + 1,
                    )
                )
            else:
                lines.append(
                    f"{indent}- {_display_value(item)}"
                )

    else:
        lines.append(
            f"{indent}- {_display_value(value)}"
        )

    return lines


def _render_localization(
    localization: Any,
) -> list[str]:
    """Render localization results as natural-language sentences."""

    if not isinstance(localization, list):
        return _render_nested(localization)

    lines: list[str] = []

    for index, item in enumerate(localization, start=1):
        if not isinstance(item, dict):
            lines.append(
                f"- {_display_value(item)}"
            )
            continue

        region = item.get(
            "region",
            str(index),
        )

        distortion = _display_value(
            item.get(
                "distortion",
                item.get("distortion_raw"),
            )
        )

        scope = item.get("scope")
        bbox = item.get("bbox")

        sentence = (
            f"- **区域 {region}**：检测到"
            f" **{distortion}**"
        )

        if scope:
            sentence += (
                f"，影响范围为"
                f" **{_display_value(scope)}**"
            )

        if (
            isinstance(bbox, list)
            and len(bbox) == 4
        ):
            sentence += (
                f"，定位坐标为 "
                f"`({bbox[0]}, {bbox[1]})"
                f" → ({bbox[2]}, {bbox[3]})`"
            )

        sentence += "。"
        lines.append(sentence)

    return lines



def _render_attribution(
    attribution: Any,
) -> list[str]:
    """Render attribution without exposing parser-internal English text."""

    if not isinstance(attribution, dict):
        return _render_nested(attribution)

    lines: list[str] = []

    count = attribution.get("degradation_count")
    primary = attribution.get("primary_impairment")

    if count is not None:
        lines.append(
            f"- 共识别到 **{_display_value(count)} 种主要退化**。"
        )

    if primary:
        primary_text = (
            str(primary)
            .strip()
            .replace("_", " ")
            .replace("-", " ")
        )

        scope = ""

        if primary_text.lower().startswith("global "):
            scope = "全局"
            primary_text = primary_text[7:]
        elif primary_text.lower().startswith("localized "):
            scope = "局部"
            primary_text = primary_text[10:]
        elif primary_text.lower().startswith("local "):
            scope = "局部"
            primary_text = primary_text[6:]

        distortion = _display_value(
            primary_text.replace(" ", "_")
        )

        if scope:
            lines.append(
                f"- 主导失真为 **{distortion}**，"
                f"主要影响范围为 **{scope}**。"
            )
        else:
            lines.append(
                f"- 主导失真为 **{distortion}**。"
            )

    try:
        degradation_count = int(count)
    except (TypeError, ValueError):
        degradation_count = 0

    if degradation_count > 1:
        lines.append(
            "- 多种退化可能相互叠加，从而进一步降低整体视觉质量。"
        )

    if not lines:
        raw_text = attribution.get("text")

        if raw_text:
            lines.append(f"- {raw_text}")

    return lines


def _render_diagnosis(
    diagnosis: Any,
) -> list[str]:
    """Render diagnosis rows once in a concise readable form."""

    if not isinstance(diagnosis, dict):
        return _render_nested(diagnosis)

    rows = diagnosis.get("rows")

    if not isinstance(rows, list):
        rows = []

        dimensions = diagnosis.get("dimensions", {})

        if isinstance(dimensions, dict):
            for dimension, details in dimensions.items():
                if isinstance(details, dict):
                    rows.append(
                        {
                            "dimension": dimension,
                            **details,
                        }
                    )

    lines: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        dimension = _display_dimension(
            row.get("dimension", "未知维度")
        )

        score = row.get("score")
        description = row.get("description")

        lines.append(
            f"- **{dimension}**："
            f"**{_display_value(score)}**"
        )

        if description:
            lines.append(
                f"  - 说明：{_translate_description(description)}"
            )

    mean_score = diagnosis.get("mean_score")

    if mean_score is None:
        mean_score = diagnosis.get("diagnosis_mean")

    if mean_score is None:
        mean_score = diagnosis.get("calculated_mean")

    if mean_score is not None:
        lines.append(
            f"- **平均失真分数**："
            f"**{_display_value(mean_score)}**"
        )

    if not lines:
        return _render_nested(diagnosis)

    return lines


def _render_restoration(
    suggestions: Any,
) -> list[str]:
    """Render restoration suggestions as readable actions."""

    if not isinstance(suggestions, list):
        return _render_nested(suggestions)

    lines: list[str] = []

    for index, item in enumerate(suggestions, start=1):
        if not isinstance(item, dict):
            lines.append(
                f"- {_display_value(item)}"
            )
            continue

        region = item.get(
            "region",
            str(index),
        )

        distortion = _display_value(
            item.get(
                "distortion",
                item.get("distortion_raw"),
            )
        )

        action = _display_value(
            item.get("action")
        )

        lines.append(
            f"- **区域 {region}**：针对"
            f" **{distortion}**，建议"
            f" **{action}**。"
        )

    return lines


def _render_expert_routing(
    routing: Any,
) -> list[str]:
    """Render expert routing without exposing raw dictionaries."""

    if not isinstance(routing, dict):
        return _render_nested(routing)

    lines: list[str] = []

    selected = routing.get("selected_expert")

    if selected:
        lines.append(
            f"- 系统最终选择："
            f"**{_display_expert(selected)}**。"
        )

    top_experts = routing.get("top_experts")

    if isinstance(top_experts, list) and top_experts:
        readable = "、".join(
            _display_expert(item)
            for item in top_experts
        )

        lines.append(
            f"- 优先候选专家：**{readable}**。"
        )

    weights = routing.get("weights", {})

    if isinstance(weights, dict) and weights:
        lines.append("- 各专家路由权重：")

        sorted_weights = sorted(
            weights.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        for name, weight in sorted_weights:
            try:
                percentage = float(weight) * 100
                weight_text = f"{percentage:.1f}%"
            except (TypeError, ValueError):
                weight_text = _display_value(weight)

            lines.append(
                f"  - {_display_expert(name)}："
                f"**{weight_text}**"
            )

    return lines


def _render_quality_prediction(
    prediction: Any,
) -> list[str]:
    """Render the predicted quality score."""

    if not isinstance(prediction, dict):
        return _render_nested(prediction)

    predicted_mos = prediction.get("predicted_mos")
    mos_scale = prediction.get("mos_scale")
    normalized_mos = prediction.get("normalized_mos")

    lines: list[str] = []

    if predicted_mos is not None:
        if mos_scale is not None:
            lines.append(
                f"- 模型预测质量分数为 "
                f"**{_display_value(predicted_mos)} / "
                f"{_display_value(mos_scale)}**。"
            )
        else:
            lines.append(
                f"- 模型预测 MOS 为 "
                f"**{_display_value(predicted_mos)}**。"
            )

    if normalized_mos is not None:
        lines.append(
            f"- 归一化质量分数为 "
            f"**{_display_value(normalized_mos)}**。"
        )

    if not lines:
        lines.extend(
            _render_nested(prediction)
        )

    return lines


def _format_single_summary(
    result: dict[str, Any],
) -> str:
    """Create a readable Chinese summary for one-image analysis."""

    lines = [
        "# 单图质量分析结果",
        "",
        (
            f"- 请求编号："
            f"`{result.get('request_id', 'N/A')}`"
        ),
        (
            f"- 工作流路由："
            f"`{result.get('route', 'N/A')}`"
        ),
    ]

    cot_result = result.get(
        "cot_iqa_result",
        {},
    )

    parsed = cot_result.get(
        "parsed_output",
        {},
    )

    localization = parsed.get(
        "localization",
        [],
    )

    quality_prediction = parsed.get(
        "quality_prediction",
        {},
    )

    expert_routing = parsed.get(
        "expert_routing",
        {},
    )

    distortions: list[str] = []

    if isinstance(localization, list):
        for item in localization:
            if not isinstance(item, dict):
                continue

            value = item.get(
                "distortion",
                item.get("distortion_raw"),
            )

            readable = _display_value(value)

            if (
                readable
                and readable not in distortions
                and readable != "未提供"
            ):
                distortions.append(readable)

    lines.extend(
        [
            "",
            "## 总体结论",
        ]
    )

    if distortions:
        lines.append(
            "- 模型识别到的主要质量问题包括："
            f"**{'、'.join(distortions)}**。"
        )
    else:
        lines.append(
            "- 模型未识别到明确的主要失真类型。"
        )

    if isinstance(quality_prediction, dict):
        predicted_mos = quality_prediction.get(
            "predicted_mos"
        )
        mos_scale = quality_prediction.get(
            "mos_scale"
        )

        if predicted_mos is not None:
            if mos_scale is not None:
                lines.append(
                    f"- 综合质量预测为 "
                    f"**{_display_value(predicted_mos)} / "
                    f"{_display_value(mos_scale)}**。"
                )
            else:
                lines.append(
                    f"- 综合质量预测 MOS 为 "
                    f"**{_display_value(predicted_mos)}**。"
                )

    if isinstance(expert_routing, dict):
        selected = expert_routing.get(
            "selected_expert"
        )

        if selected:
            lines.append(
                f"- 系统建议优先调用"
                f" **{_display_expert(selected)}**。"
            )

    pyiqa = (
        result.get("pyiqa_results", {})
        .get("target_image", {})
        .get("results", {})
    )

    if pyiqa:
        lines.extend(
            [
                "",
                "## 客观质量指标",
            ]
        )

        for metric_name, metric_data in pyiqa.items():
            score = metric_data.get("score")
            lower_better = metric_data.get(
                "lower_better"
            )

            direction = (
                "数值越低通常表示质量越好"
                if lower_better
                else "数值越高通常表示质量越好"
            )

            lines.append(
                f"- **{metric_name.upper()}**："
                f"**{_display_value(score)}**；"
                f"{direction}。"
            )

    if parsed:
        lines.extend(
            [
                "",
                "## CoT-IQA 结构化诊断",
            ]
        )

        section_renderers = [
            (
                "localization",
                "失真定位",
                _render_localization,
            ),
            (
                "attribution",
                "失真归因",
                _render_attribution,
            ),
            (
                "diagnosis",
                "质量诊断",
                _render_diagnosis,
            ),
            (
                "restoration_suggestion",
                "修复建议",
                _render_restoration,
            ),
            (
                "expert_routing",
                "专家路由",
                _render_expert_routing,
            ),
            (
                "quality_prediction",
                "质量预测",
                _render_quality_prediction,
            ),
        ]

        for key, title, renderer in section_renderers:
            value = parsed.get(key)

            if not value:
                continue

            lines.extend(
                [
                    "",
                    f"### {title}",
                ]
            )

            lines.extend(
                renderer(value)
            )

    rag_context = result.get(
        "rag_context",
        [],
    )

    if rag_context:
        lines.extend(
            [
                "",
                "## 检索到的论文依据",
                *_format_rag_sources(rag_context),
            ]
        )

    verification = result.get(
        "verification_result",
        {},
    )

    if verification:
        lines.extend(
            [
                "",
                "## 结果验证",
                (
                    f"- 验证状态："
                    f"**{_display_value(verification.get('status'))}**"
                ),
                (
                    f"- 通过检查："
                    f"**{verification.get('passed_count', 0)}**"
                ),
                (
                    f"- 警告数量："
                    f"**{verification.get('warning_count', 0)}**"
                ),
                (
                    f"- 失败数量："
                    f"**{verification.get('failed_count', 0)}**"
                ),
            ]
        )

        warnings = [
            warning
            for warning in verification.get(
                "warnings",
                [],
            )
            if str(warning).strip()
        ]

        if warnings:
            lines.append("- 验证警告：")

            for warning in warnings:
                lines.append(
                    f"  - {_translate_warning(warning)}"
                )

    errors = result.get("errors", [])

    if errors:
        lines.extend(
            [
                "",
                "## 工作流错误",
            ]
        )

        for error in errors:
            if isinstance(error, dict):
                node = error.get("node", "unknown")
                message = error.get(
                    "message",
                    error.get("error", str(error)),
                )

                lines.append(
                    f"- {node}：{message}"
                )
            else:
                lines.append(
                    f"- {str(error)}"
                )

    return "\n".join(lines)



def _translate_comparison_reason(
    reason: Any,
) -> str:
    """Translate comparison decision rationale into Chinese."""

    text_value = str(reason).strip()

    prefix = (
        "All available PyIQA metrics agree that "
    )
    suffix = " has better quality."

    if (
        text_value.startswith(prefix)
        and text_value.endswith(suffix)
    ):
        item_id = text_value[
            len(prefix):-len(suffix)
        ]

        return (
            "所有可用的 PyIQA 指标均认为"
            f" {_display_value(item_id)} 的质量更好。"
        )

    prefix = (
        "Structured CoT diagnosis favors "
    )
    suffix = (
        " because its mean distortion score "
        "is lower."
    )

    if (
        text_value.startswith(prefix)
        and text_value.endswith(suffix)
    ):
        item_id = text_value[
            len(prefix):-len(suffix)
        ]

        return (
            "结构化 CoT 诊断认为"
            f" {_display_value(item_id)} 的质量更好，"
            "因为其平均失真分数更低。"
        )

    marker = " receives "
    suffix = " primary evidence vote(s)."

    if (
        marker in text_value
        and text_value.endswith(suffix)
    ):
        item_id, vote_text = text_value.split(
            marker,
            1,
        )

        vote_text = vote_text[
            :-len(suffix)
        ].replace(" of ", "/")

        return (
            f"{_display_value(item_id)} 获得了"
            f" **{vote_text}** 个主证据投票。"
        )

    return (
        text_value
        .replace("image_1", "图片 1")
        .replace("image_2", "图片 2")
    )


def _format_comparison_summary(
    result: dict[str, Any],
) -> str:
    """Create a readable Chinese summary for two-image comparison."""

    comparison = result.get(
        "comparison_result",
        {},
    )

    winner = comparison.get(
        "winner_item_id",
        comparison.get("winner", "N/A"),
    )

    winner_text = _display_value(winner)

    lines = [
        "# 双图质量比较结果",
        "",
        (
            f"- 请求编号："
            f"`{result.get('request_id', 'N/A')}`"
        ),
        (
            f"- 工作流路由："
            f"`{result.get('route', 'N/A')}`"
        ),
        "",
        "## 最终结论",
        (
            f"- 综合 PyIQA 指标和结构化诊断，"
            f"**{winner_text} 的整体质量更好**。"
        ),
        (
            f"- 决策置信度："
            f"**{_display_value(comparison.get('confidence'))}**。"
        ),
        (
            f"- 决策状态："
            f"**{_display_value(comparison.get('status'))}**。"
        ),
    ]

    votes = comparison.get(
        "trusted_vote_counts",
        {},
    )

    if votes:
        lines.extend(
            [
                "",
                "## 主证据投票",
            ]
        )

        for item_id, count in votes.items():
            lines.append(
                f"- **{_display_value(item_id)}**："
                f"**{count} 票**"
            )

    evidence = comparison.get(
        "evidence",
        {},
    )

    pyiqa_evidence = evidence.get(
        "pyiqa",
        {},
    )

    if pyiqa_evidence:
        lines.extend(
            [
                "",
                "## 客观指标比较",
            ]
        )

        for metric_name, metric_data in pyiqa_evidence.items():
            first_score = metric_data.get(
                "first_score"
            )

            second_score = metric_data.get(
                "second_score"
            )

            metric_winner = metric_data.get(
                "winner_item_id"
            )

            lines.append(
                f"- **{metric_name.upper()}**："
                f"图片 1 为 **{_display_value(first_score)}**，"
                f"图片 2 为 **{_display_value(second_score)}**；"
                f"该指标认为"
                f" **{_display_value(metric_winner)}** 更好。"
            )

    diagnosis = evidence.get(
        "diagnosis",
        {},
    )

    if diagnosis:
        first_mean = diagnosis.get(
            "first_mean_score"
        )
        second_mean = diagnosis.get(
            "second_mean_score"
        )
        diagnosis_winner = diagnosis.get(
            "winner_item_id"
        )

        lines.extend(
            [
                "",
                "## 结构化诊断比较",
                (
                    f"- 图片 1 平均失真分数："
                    f"**{_display_value(first_mean)}**"
                ),
                (
                    f"- 图片 2 平均失真分数："
                    f"**{_display_value(second_mean)}**"
                ),
            ]
        )

        if diagnosis_winner:
            lines.append(
                "- 结构化失真诊断认为"
                f" **{_display_value(diagnosis_winner)}**"
                " 的质量更好。"
            )
        elif (
            isinstance(first_mean, (int, float))
            and isinstance(second_mean, (int, float))
            and abs(first_mean - second_mean) < 1e-8
        ):
            lines.append(
                "- 两张图片的结构化失真分数相同，"
                "本项不参与主证据投票。"
            )
        else:
            lines.append(
                "- 当前结构化诊断未形成明确的优劣结论，"
                "本项不参与主证据投票。"
            )

    generated_mos = evidence.get(
        "generated_mos",
        {},
    )

    if generated_mos:
        first_mos = generated_mos.get(
            "first_mos"
        )
        second_mos = generated_mos.get(
            "second_mos"
        )
        mos_winner = generated_mos.get(
            "winner_item_id"
        )

        lines.extend(
            [
                "",
                "## 模型生成 MOS（辅助证据）",
                (
                    f"- 图片 1 MOS："
                    f"**{_display_value(first_mos)}**"
                ),
                (
                    f"- 图片 2 MOS："
                    f"**{_display_value(second_mos)}**"
                ),
            ]
        )

        if (
            isinstance(first_mos, (int, float))
            and isinstance(second_mos, (int, float))
            and abs(first_mos - second_mos) < 1e-8
        ):
            lines.append(
                "- 两张图片的模型生成 MOS 相同，"
                "该项无法区分优劣。"
            )
        elif mos_winner:
            lines.append(
                "- 模型生成 MOS 倾向于"
                f" **{_display_value(mos_winner)}**，"
                "但该结果仅作为辅助证据。"
            )
        else:
            lines.append(
                "- 当前模型生成 MOS 未形成明确结论。"
            )

        lines.append(
            "- 模型生成 MOS 不会覆盖 PyIQA 和"
            "结构化诊断形成的主决策。"
        )

    rationale = comparison.get(
        "rationale",
        [],
    )

    if rationale:
        lines.extend(
            [
                "",
                "## 判断依据",
            ]
        )

        for reason in rationale:
            lines.append(
                f"- {_translate_comparison_reason(reason)}"
            )

    conflicts = comparison.get(
        "conflicts",
        [],
    )

    if conflicts:
        lines.extend(
            [
                "",
                "## 证据冲突说明",
            ]
        )

        for conflict in conflicts:
            readable_conflict = (
                str(conflict)
                .replace(
                    "Generated MOS conflicts with the primary "
                    "PyIQA and diagnosis decision.",
                    "模型生成的 MOS 与 PyIQA 和结构化诊断的"
                    "主结论不一致。",
                )
            )

            lines.append(
                f"- {readable_conflict}"
            )

    rag_context = result.get(
        "rag_context",
        [],
    )

    if rag_context:
        lines.extend(
            [
                "",
                "## 检索到的论文依据",
                *_format_rag_sources(rag_context),
            ]
        )

    verification = result.get(
        "verification_result",
        {},
    )

    if verification:
        lines.extend(
            [
                "",
                "## 结果验证",
                (
                    f"- 验证状态："
                    f"**{_display_value(verification.get('status'))}**"
                ),
                (
                    f"- 通过检查："
                    f"**{verification.get('passed_count', 0)}**"
                ),
                (
                    f"- 警告数量："
                    f"**{verification.get('warning_count', 0)}**"
                ),
                (
                    f"- 失败数量："
                    f"**{verification.get('failed_count', 0)}**"
                ),
            ]
        )

    errors = result.get("errors", [])

    if errors:
        lines.extend(
            [
                "",
                "## 工作流错误",
            ]
        )

        for error in errors:
            if isinstance(error, dict):
                node = error.get("node", "unknown")
                message = error.get(
                    "message",
                    error.get("error", str(error)),
                )

                lines.append(
                    f"- {node}：{message}"
                )
            else:
                lines.append(
                    f"- {str(error)}"
                )

    return "\n".join(lines)


def analyze_single_image(
    image_path: str | None,
    user_query: str,
) -> tuple[str, dict[str, Any]]:
    """Run the complete single-image Agent workflow."""

    if not image_path:
        return (
            "## 输入错误\n\n请先上传一张图片。",
            {},
        )

    query = (
        user_query.strip()
        if user_query and user_query.strip()
        else DEFAULT_SINGLE_QUERY
    )

    try:
        initial_state = create_initial_state(
            user_query=query,
            image_paths=[image_path],
        )

        result = agent_app.invoke(
            initial_state,
            config={
                "recursion_limit": 30,
            },
        )

        return (
            _format_single_summary(result),
            _compact_result(result),
        )

    except Exception as exc:
        print("\n===== GRADIO WORKFLOW ERROR =====")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("===== END GRADIO WORKFLOW ERROR =====\n")

        error_data = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

        return (
            "## 运行失败\n\n"
            f"`{type(exc).__name__}: {exc}`",
            error_data,
        )


def compare_two_images(
    image_1_path: str | None,
    image_2_path: str | None,
    user_query: str,
) -> tuple[str, dict[str, Any]]:
    """Run the complete two-image Agent workflow."""

    if not image_1_path or not image_2_path:
        return (
            "## 输入错误\n\n请上传两张待比较图片。",
            {},
        )

    query = (
        user_query.strip()
        if user_query and user_query.strip()
        else DEFAULT_COMPARISON_QUERY
    )

    try:
        initial_state = create_initial_state(
            user_query=query,
            image_paths=[
                image_1_path,
                image_2_path,
            ],
        )

        result = agent_app.invoke(
            initial_state,
            config={
                "recursion_limit": 40,
            },
        )

        return (
            _format_comparison_summary(result),
            _compact_result(result),
        )

    except Exception as exc:
        print("\n===== GRADIO WORKFLOW ERROR =====")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("===== END GRADIO WORKFLOW ERROR =====\n")

        error_data = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

        return (
            "## 运行失败\n\n"
            f"`{type(exc).__name__}: {exc}`",
            error_data,
        )


def build_demo() -> gr.Blocks:
    """Build the Gradio application."""

    with gr.Blocks(
        title="COT-IQA-Agent",
    ) as demo:
        gr.Markdown(
            """
# COT-IQA-Agent

基于 **CoT-IQA、PyIQA、LangGraph 和论文 RAG**
的可解释图像质量分析系统。

支持单图质量诊断、双图质量比较、失真定位、
修复建议、专家路由、论文依据检索和结果验证。
"""
        )

        with gr.Tab("单图质量分析"):
            with gr.Row():
                single_image = gr.Image(
                    label="上传待分析图片",
                    type="filepath",
                    sources=["upload"],
                    height=420,
                )

                single_query = gr.Textbox(
                    label="分析要求",
                    value=DEFAULT_SINGLE_QUERY,
                    lines=7,
                )

            single_button = gr.Button(
                "开始单图分析",
                variant="primary",
            )

            single_summary = gr.Markdown(
                label="分析摘要",
            )

            with gr.Accordion(
                "查看完整结构化结果（调试信息）",
                open=False,
            ):
                single_json = gr.JSON(
                    label="结构化 JSON",
                )

            single_button.click(
                fn=analyze_single_image,
                inputs=[
                    single_image,
                    single_query,
                ],
                outputs=[
                    single_summary,
                    single_json,
                ],
                concurrency_limit=1,
                concurrency_id="cot_iqa_gpu",
                api_name="analyze_single_image",
            )

            gr.ClearButton(
                [
                    single_image,
                    single_query,
                    single_summary,
                    single_json,
                ]
            )

        with gr.Tab("双图质量比较"):
            with gr.Row():
                comparison_image_1 = gr.Image(
                    label="图片 1",
                    type="filepath",
                    sources=["upload"],
                    height=360,
                )

                comparison_image_2 = gr.Image(
                    label="图片 2",
                    type="filepath",
                    sources=["upload"],
                    height=360,
                )

            comparison_query = gr.Textbox(
                label="比较要求",
                value=DEFAULT_COMPARISON_QUERY,
                lines=5,
            )

            comparison_button = gr.Button(
                "开始双图比较",
                variant="primary",
            )

            comparison_summary = gr.Markdown(
                label="比较摘要",
            )

            with gr.Accordion(
                "查看完整结构化结果（调试信息）",
                open=False,
            ):
                comparison_json = gr.JSON(
                    label="结构化 JSON",
                )

            comparison_button.click(
                fn=compare_two_images,
                inputs=[
                    comparison_image_1,
                    comparison_image_2,
                    comparison_query,
                ],
                outputs=[
                    comparison_summary,
                    comparison_json,
                ],
                concurrency_limit=1,
                concurrency_id="cot_iqa_gpu",
                api_name="compare_two_images",
            )

            gr.ClearButton(
                [
                    comparison_image_1,
                    comparison_image_2,
                    comparison_query,
                    comparison_summary,
                    comparison_json,
                ]
            )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(
        max_size=8,
        default_concurrency_limit=1,
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
