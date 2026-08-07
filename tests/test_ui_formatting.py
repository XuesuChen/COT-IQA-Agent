"""Readable Gradio-output formatting tests."""

from ui.gradio_app import (
    _format_comparison_summary,
    _format_single_summary,
)


def test_single_summary_is_readable_chinese():
    result = {
        "request_id": "req_test123",
        "route": "single_image",
        "pyiqa_results": {
            "target_image": {
                "results": {
                    "niqe": {
                        "score": 2.83,
                        "lower_better": True,
                    }
                }
            }
        },
        "cot_iqa_result": {
            "parsed_output": {
                "localization": [
                    {
                        "region": "A",
                        "bbox": [0, 0, 512, 512],
                        "scope": "global",
                        "distortion": (
                            "jpeg_compression"
                        ),
                    }
                ],
                "attribution": {
                    "degradation_count": 1,
                    "primary_impairment": (
                        "global_jpeg_compression"
                    ),
                },
                "diagnosis": {
                    "rows": [
                        {
                            "dimension": "color",
                            "score": 0.3,
                            "description": (
                                "Color appears accurate "
                                "and natural"
                            ),
                        },
                        {
                            "dimension": "artifact",
                            "score": 6.7,
                            "description": (
                                "Significant "
                                "multi-degradation "
                                "interaction artifacts"
                            ),
                        },
                    ],
                    "mean_score": 3.5,
                },
                "restoration_suggestion": [
                    {
                        "region": "A",
                        "distortion": (
                            "jpeg_compression"
                        ),
                        "action": (
                            "apply_compression_"
                            "artifact_reduction"
                        ),
                    }
                ],
                "expert_routing": {
                    "selected_expert": "artifact",
                    "top_experts": ["artifact"],
                    "weights": {
                        "artifact": 0.7,
                    },
                },
                "quality_prediction": {
                    "predicted_mos": 2.0,
                    "mos_scale": 5.0,
                },
            }
        },
        "verification_result": {
            "status": "ok",
            "passed_count": 11,
            "warning_count": 0,
            "failed_count": 0,
            "warnings": [],
        },
        "errors": [],
    }

    output = _format_single_summary(result)

    assert "JPEG 压缩伪影" in output
    assert "颜色表现" in output
    assert "综合伪影" in output
    assert "通用伪影修复专家" in output
    assert "全局" in output
    assert "```json" not in output
    assert "dimension raw" not in output.lower()


def test_comparison_tie_summary():
    result = {
        "request_id": "req_tie123",
        "route": "comparison",
        "comparison_result": {
            "status": "warning",
            "winner_item_id": "image_1",
            "confidence": "medium",
            "trusted_vote_counts": {
                "image_1": 2,
                "image_2": 0,
            },
            "evidence": {
                "diagnosis": {
                    "first_mean_score": 2.66,
                    "second_mean_score": 2.66,
                    "winner_item_id": None,
                },
                "generated_mos": {
                    "first_mos": 2.0,
                    "second_mos": 2.0,
                    "winner_item_id": None,
                },
            },
            "rationale": [
                (
                    "image_1 receives 2 of 2 "
                    "primary evidence vote(s)."
                )
            ],
            "conflicts": [],
        },
        "verification_result": {
            "status": "warning",
            "passed_count": 6,
            "warning_count": 1,
            "failed_count": 0,
        },
        "errors": [],
    }

    output = _format_comparison_summary(result)

    assert "结构化失真分数相同" in output
    assert "模型生成 MOS 相同" in output
    assert "本项不参与主证据投票" in output
    assert "2/2" in output
    assert "未提供 质量更好" not in output


def test_single_summary_accepts_string_errors():
    """String-form workflow errors must not crash the UI formatter."""

    result = {
        "request_id": "req_error123",
        "route": "single_image",
        "errors": [
            (
                "CoT-IQA loading or inference failed: "
                "TypeError: unhashable type: 'set'"
            )
        ],
    }

    output = _format_single_summary(result)

    assert "工作流错误" in output
    assert "CoT-IQA loading or inference failed" in output
    assert "unhashable type" in output


def test_localized_primary_impairment_translation():
    """Parser attribution using 'localized' must render in Chinese."""

    from ui.gradio_app import _render_attribution

    lines = _render_attribution(
        {
            "degradation_count": 2,
            "primary_impairment": (
                "localized jpeg compression"
            ),
        }
    )

    output = "\n".join(lines)

    assert "2 种主要退化" in output
    assert "JPEG 压缩伪影" in output
    assert "局部" in output
    assert "localized jpeg compression" not in output
