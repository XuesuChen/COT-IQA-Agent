"""FastAPI contract tests without loading AI models."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import app as api_module


def _make_png() -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        (32, 32),
        (128, 128, 128),
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


async def _fake_invoke_workflow(
    initial_state,
    *,
    recursion_limit,
):
    image_count = len(
        initial_state.get("image_paths", [])
    )

    route = (
        "comparison"
        if image_count == 2
        else "single_image"
    )

    result = {
        **initial_state,
        "route": route,
        "execution_trace": [
            "state_initialized",
            f"router:{route}",
            "mock_completed",
        ],
        "rag_context": [
            {
                "rank": 1,
                "score": 0.81,
                "text": "This full text must not be public.",
                "metadata": {
                    "source_path": "mock/paper.pdf",
                    "page_number": 2,
                    "title": "Mock IQA Paper",
                },
                "citation": "mock/paper.pdf#page=2",
            }
        ],
        "verification_result": {
            "status": "ok",
            "check_count": 3,
            "passed_count": 3,
            "warning_count": 0,
            "failed_count": 0,
            "warnings": [],
            "errors": [],
        },
        "errors": [],
        "final_report": (
            "Large report text must not appear."
        ),
    }

    if route == "single_image":
        result.update(
            {
                "pyiqa_results": {
                    "target_image": {
                        "results": {
                            "niqe": {
                                "score": 2.8,
                                "lower_better": True,
                            },
                            "brisque": {
                                "score": 19.5,
                                "lower_better": True,
                            },
                        }
                    }
                },
                "cot_iqa_result": {
                    "raw_output": (
                        "Raw CoT must not appear."
                    ),
                    "parsed_output": {
                        "localization": [
                            {
                                "region": "A",
                                "bbox": [0, 0, 32, 32],
                                "scope": "global",
                                "distortion": (
                                    "gaussian_blur"
                                ),
                                "severity": "mild",
                            }
                        ],
                        "diagnosis": {
                            "mean_score": 2.6,
                        },
                        "quality_prediction": {
                            "predicted_mos": 2.0,
                            "mos_scale": 5.0,
                            "normalized_mos": 0.4,
                        },
                        "expert_routing": {
                            "selected_expert": "deblur",
                            "weights": {
                                "deblur": 0.8,
                            },
                        },
                        "restoration_suggestion": [
                            {
                                "region": "A",
                                "distortion": (
                                    "gaussian_blur"
                                ),
                                "action": (
                                    "apply_deblurring"
                                ),
                            }
                        ],
                    },
                },
            }
        )

    else:
        result["comparison_result"] = {
            "status": "ok",
            "winner_item_id": "image_1",
            "loser_item_id": "image_2",
            "confidence": "high",
            "decision_basis": (
                "pyiqa_metrics_and_structured_diagnosis"
            ),
            "trusted_vote_counts": {
                "image_1": 3,
                "image_2": 0,
            },
            "evidence": {
                "pyiqa": {
                    "niqe": {
                        "winner_item_id": "image_1",
                    }
                },
                "diagnosis": {
                    "winner_item_id": "image_1",
                },
                "generated_mos": {},
            },
            "rationale": [
                "image_1 is better."
            ],
            "conflicts": [],
        }

    return result


def test_single_image_api(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "_invoke_workflow",
        _fake_invoke_workflow,
    )

    client = TestClient(api_module.app)

    response = client.post(
        "/api/v1/analyze",
        files={
            "image": (
                "single.png",
                _make_png(),
                "image/png",
            )
        },
        data={
            "query": "测试单图分析",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["route"] == "single_image"
    assert data["request_id"]
    assert data["summary"]["predicted_mos"] == 2.0
    assert (
        data["summary"]["selected_expert"]
        == "deblur"
    )
    assert len(data["rag_sources"]) == 1
    assert data["report_urls"]["view"].endswith(
        "/view"
    )
    assert data["errors"] == []

    response_text = response.text.lower()

    assert "raw_output" not in response_text
    assert "final_report" not in response_text
    assert "full text must not be public" not in (
        response_text
    )


def test_comparison_api(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "_invoke_workflow",
        _fake_invoke_workflow,
    )

    client = TestClient(api_module.app)
    image = _make_png()

    response = client.post(
        "/api/v1/compare",
        files={
            "image_1": (
                "image_1.png",
                image,
                "image/png",
            ),
            "image_2": (
                "image_2.png",
                image,
                "image/png",
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["route"] == "comparison"
    assert (
        data["summary"]["winner_item_id"]
        == "image_1"
    )
    assert data["summary"]["confidence"] == "high"
    assert data["verification"]["failed_count"] == 0
    assert data["errors"] == []


def test_invalid_upload_format(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "_invoke_workflow",
        _fake_invoke_workflow,
    )

    client = TestClient(api_module.app)

    response = client.post(
        "/api/v1/analyze",
        files={
            "image": (
                "invalid.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
