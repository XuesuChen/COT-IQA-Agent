"""Generated-report endpoint tests."""

import json

from fastapi.testclient import TestClient

import app as api_module


def test_report_read_view_and_download(
    tmp_path,
    monkeypatch,
):
    request_id = "req_abc123"
    report_type = "single_image"

    json_path = (
        tmp_path
        / f"{request_id}_{report_type}.json"
    )

    markdown_path = (
        tmp_path
        / f"{request_id}_{report_type}.md"
    )

    json_path.write_text(
        json.dumps(
            {
                "request_id": request_id,
                "status": "ok",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    markdown_path.write_text(
        "# 测试报告\n\n这是可读报告内容。",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        api_module,
        "_get_report_directory",
        lambda: tmp_path,
    )

    client = TestClient(api_module.app)

    content_response = client.get(
        f"/api/v1/reports/{request_id}"
    )

    assert content_response.status_code == 200

    content = content_response.json()

    assert content["status"] == "ok"
    assert content["report_type"] == report_type
    assert content["json_report"]["status"] == "ok"
    assert "测试报告" in content["markdown_report"]

    view_response = client.get(
        f"/api/v1/reports/{request_id}/view"
    )

    assert view_response.status_code == 200
    assert "text/html" in view_response.headers[
        "content-type"
    ]
    assert "COT-IQA-Agent 分析报告" in (
        view_response.text
    )
    assert "测试报告" in view_response.text

    json_download = client.get(
        f"/api/v1/reports/{request_id}"
        "/download/json"
    )

    assert json_download.status_code == 200
    assert "application/json" in (
        json_download.headers["content-type"]
    )

    markdown_download = client.get(
        f"/api/v1/reports/{request_id}"
        "/download/markdown"
    )

    assert markdown_download.status_code == 200
    assert "text/markdown" in (
        markdown_download.headers["content-type"]
    )


def test_report_errors(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "_get_report_directory",
        lambda: tmp_path,
    )

    client = TestClient(api_module.app)

    missing = client.get(
        "/api/v1/reports/req_missing123"
    )

    invalid_id = client.get(
        "/api/v1/reports/invalid-id"
    )

    assert missing.status_code == 404
    assert invalid_id.status_code == 400
