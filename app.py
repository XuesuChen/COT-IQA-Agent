"""FastAPI service entrypoint for COT-IQA-Agent."""

from __future__ import annotations

import asyncio
import html
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import bleach
import markdown as markdown_lib
import torch
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from agent.graph import app as agent_graph
from agent.state import create_initial_state
from configs.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = load_config()


def _resolve_project_path(
    value: str | Path | None,
) -> Path | None:
    """Resolve a configured project path."""

    if value is None:
        return None

    path = Path(str(value)).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _get_config_value(
    *keys: str,
    default: Any = None,
) -> Any:
    """Read a nested configuration value safely."""

    value: Any = CONFIG

    for key in keys:
        if not isinstance(value, dict):
            return default

        value = value.get(key)

        if value is None:
            return default

    return value


PROJECT_NAME = str(
    _get_config_value(
        "project",
        "name",
        default="COT-IQA-Agent",
    )
)

PROJECT_VERSION = str(
    _get_config_value(
        "project",
        "version",
        default="0.1.0",
    )
)


class HealthResponse(BaseModel):
    """Health-check response schema."""

    status: str
    service: str
    version: str
    environment: str
    cuda_available: bool
    cuda_device_name: str | None
    cot_iqa_model_configured: bool
    cot_iqa_base_model_path: str | None
    cot_iqa_adapter_path: str | None
    rag_enabled: bool
    rag_index_ready: bool
    rag_vector_store: str | None
    report_directory_ready: bool
    report_directory: str | None


app = FastAPI(
    title="COT-IQA-Agent API",
    version=PROJECT_VERSION,
    description=(
        "REST API for explainable single-image quality analysis, "
        "two-image quality comparison, paper RAG retrieval, "
        "verification, and report generation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["service"],
)
def root() -> dict[str, Any]:
    """Return API navigation information."""

    return {
        "service": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "status": "running",
        "health": "/health",
        "swagger": "/docs",
        "redoc": "/redoc",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["service"],
)
def health_check() -> HealthResponse:
    """Check service resources without loading AI models."""

    base_model_path = _resolve_project_path(
        _get_config_value(
            "cot_iqa",
            "base_model_path",
        )
    )

    adapter_path = _resolve_project_path(
        _get_config_value(
            "cot_iqa",
            "adapter_path",
        )
    )

    base_model_ready = bool(
        base_model_path
        and base_model_path.is_dir()
        and (base_model_path / "config.json").is_file()
    )

    adapter_ready = bool(
        adapter_path
        and adapter_path.is_dir()
        and (adapter_path / "adapter_config.json").is_file()
    )

    vector_store_path = _resolve_project_path(
        _get_config_value(
            "paths",
            "vector_store_dir",
            default=_get_config_value(
                "rag",
                "vector_store_dir",
            ),
        )
    )

    report_directory = _resolve_project_path(
        _get_config_value(
            "paths",
            "report_dir",
            default="outputs/reports",
        )
    )

    required_rag_files = (
        "index.faiss",
        "chunks.jsonl",
        "manifest.json",
    )

    rag_index_ready = bool(
        vector_store_path
        and vector_store_path.is_dir()
        and all(
            (vector_store_path / file_name).is_file()
            for file_name in required_rag_files
        )
    )

    report_directory_ready = bool(
        report_directory
        and report_directory.is_dir()
    )

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        cuda_device_name = torch.cuda.get_device_name(0)
    else:
        cuda_device_name = None

    return HealthResponse(
        status="ok",
        service=PROJECT_NAME,
        version=PROJECT_VERSION,
        environment=str(
            _get_config_value(
                "project",
                "environment",
                default="development",
            )
        ),
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        cot_iqa_model_configured=(
            base_model_ready
            and adapter_ready
        ),
        cot_iqa_base_model_path=(
            str(base_model_path)
            if base_model_path is not None
            else None
        ),
        cot_iqa_adapter_path=(
            str(adapter_path)
            if adapter_path is not None
            else None
        ),
        rag_enabled=bool(
            _get_config_value(
                "rag",
                "enabled",
                default=True,
            )
        ),
        rag_index_ready=rag_index_ready,
        rag_vector_store=(
            str(vector_store_path)
            if vector_store_path is not None
            else None
        ),
        report_directory_ready=(
            report_directory_ready
        ),
        report_directory=(
            str(report_directory)
            if report_directory is not None
            else None
        ),
    )



DEFAULT_SINGLE_QUERY = (
    "请分析这张图片的整体质量，识别主要失真、严重程度和影响区域，"
    "给出修复建议，并结合相关 IQA 论文知识进行解释。"
)

DEFAULT_COMPARISON_QUERY = (
    "请比较这两张图片的整体质量，识别各自的主要失真，"
    "判断哪张图片质量更好，并结合相关 IQA 论文知识解释判断依据。"
)

ALLOWED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_INFERENCE_LOCK = asyncio.Lock()


class AgentResponse(BaseModel):
    """Compact public response for Agent workflows."""

    status: str
    request_id: str | None
    route: str | None
    summary: dict[str, Any]
    verification: dict[str, Any]
    rag_sources: list[dict[str, Any]]
    report_urls: dict[str, str]
    execution_trace: list[str]
    errors: list[Any]


def _json_safe(value: Any) -> Any:
    """Convert workflow output into JSON-compatible data."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    )


def _compact_rag_sources(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return paper references without full chunk text."""

    sources: list[dict[str, Any]] = []

    for item in result.get("rag_context", []):
        metadata = item.get("metadata", {})

        sources.append(
            {
                "rank": item.get("rank"),
                "score": item.get("score"),
                "source_path": metadata.get(
                    "source_path"
                ),
                "page_number": metadata.get(
                    "page_number"
                ),
                "title": metadata.get("title"),
                "citation": item.get("citation"),
            }
        )

    return _json_safe(sources)


def _compact_verification(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return the public verification summary."""

    verification = result.get(
        "verification_result",
        {},
    )

    return _json_safe(
        {
            "status": verification.get("status"),
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
            "warnings": verification.get(
                "warnings",
                [],
            ),
            "errors": verification.get(
                "errors",
                [],
            ),
        }
    )


def _extract_single_summary(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build the compact single-image response."""

    metric_results = (
        result.get("pyiqa_results", {})
        .get("target_image", {})
        .get("results", {})
    )

    pyiqa_metrics: dict[str, Any] = {}

    for metric_name, metric_data in (
        metric_results.items()
    ):
        pyiqa_metrics[metric_name] = {
            "score": metric_data.get("score"),
            "lower_better": metric_data.get(
                "lower_better"
            ),
        }

    parsed = (
        result.get("cot_iqa_result", {})
        .get("parsed_output", {})
    )

    localization = parsed.get(
        "localization",
        [],
    )

    detected_distortions: list[dict[str, Any]] = []

    if isinstance(localization, list):
        for item in localization:
            if not isinstance(item, dict):
                continue

            detected_distortions.append(
                {
                    "region": item.get("region"),
                    "bbox": item.get("bbox"),
                    "scope": item.get("scope"),
                    "distortion": item.get(
                        "distortion",
                        item.get("distortion_raw"),
                    ),
                    "severity": item.get(
                        "severity"
                    ),
                }
            )

    diagnosis = parsed.get(
        "diagnosis",
        {},
    )

    diagnosis_mean = diagnosis.get(
        "mean_score"
    )

    if diagnosis_mean is None:
        diagnosis_mean = diagnosis.get(
            "diagnosis_mean"
        )

    if diagnosis_mean is None:
        diagnosis_mean = diagnosis.get(
            "calculated_mean"
        )

    quality = parsed.get(
        "quality_prediction",
        {},
    )

    routing = parsed.get(
        "expert_routing",
        {},
    )

    return _json_safe(
        {
            "task": "single_image",
            "pyiqa_metrics": pyiqa_metrics,
            "detected_distortions": (
                detected_distortions
            ),
            "diagnosis_mean": diagnosis_mean,
            "predicted_mos": quality.get(
                "predicted_mos"
            ),
            "mos_scale": quality.get(
                "mos_scale"
            ),
            "normalized_mos": quality.get(
                "normalized_mos"
            ),
            "selected_expert": routing.get(
                "selected_expert"
            ),
            "expert_weights": routing.get(
                "weights",
                {},
            ),
            "restoration_suggestions": (
                parsed.get(
                    "restoration_suggestion",
                    [],
                )
            ),
        }
    )


def _extract_comparison_summary(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build the compact two-image response."""

    comparison = result.get(
        "comparison_result",
        {},
    )

    evidence = comparison.get(
        "evidence",
        {},
    )

    return _json_safe(
        {
            "task": "comparison",
            "winner_item_id": comparison.get(
                "winner_item_id",
                comparison.get("winner"),
            ),
            "loser_item_id": comparison.get(
                "loser_item_id"
            ),
            "confidence": comparison.get(
                "confidence"
            ),
            "decision_status": comparison.get(
                "status"
            ),
            "decision_basis": comparison.get(
                "decision_basis"
            ),
            "trusted_vote_counts": (
                comparison.get(
                    "trusted_vote_counts",
                    {},
                )
            ),
            "pyiqa_comparison": evidence.get(
                "pyiqa",
                {},
            ),
            "diagnosis_comparison": evidence.get(
                "diagnosis",
                {},
            ),
            "generated_mos": evidence.get(
                "generated_mos",
                {},
            ),
            "rationale": comparison.get(
                "rationale",
                [],
            ),
            "conflicts": comparison.get(
                "conflicts",
                [],
            ),
        }
    )


def _build_public_summary(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Select the correct compact task summary."""

    route = result.get("route")

    if route == "comparison":
        return _extract_comparison_summary(
            result
        )

    return _extract_single_summary(result)


def _workflow_status(
    result: dict[str, Any],
) -> str:
    """Convert workflow state into a public API status."""

    errors = result.get("errors", [])

    verification = result.get(
        "verification_result",
        {},
    )

    if verification.get("failed_count", 0) > 0:
        return "failed"

    if errors:
        return "partial"

    if verification.get("warning_count", 0) > 0:
        return "warning"

    return "ok"


async def _save_upload(
    upload: UploadFile,
    directory: Path,
    label: str,
) -> Path:
    """Validate and save an uploaded image temporarily."""

    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image format. "
                f"Allowed formats: "
                f"{', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}"
            ),
        )

    destination = directory / f"{label}{suffix}"

    total_size = 0

    try:
        with destination.open("wb") as file:
            while True:
                chunk = await upload.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded image exceeds "
                            "the 20 MB size limit."
                        ),
                    )

                file.write(chunk)

    finally:
        await upload.close()

    if total_size == 0:
        destination.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    return destination.resolve()


async def _invoke_workflow(
    initial_state: dict[str, Any],
    *,
    recursion_limit: int,
) -> dict[str, Any]:
    """Run GPU workflow serially without blocking the API loop."""

    async with _INFERENCE_LOCK:
        return await asyncio.to_thread(
            agent_graph.invoke,
            initial_state,
            config={
                "recursion_limit": recursion_limit,
            },
        )


def _build_agent_response(
    result: dict[str, Any],
) -> AgentResponse:
    """Build a compact public API response."""

    request_id = result.get("request_id")

    if request_id:
        report_base = (
            f"/api/v1/reports/{request_id}"
        )

        report_urls = {
            "content": report_base,
            "view": (
                f"{report_base}/view"
            ),
            "json": (
                f"{report_base}/download/json"
            ),
            "markdown": (
                f"{report_base}/download/markdown"
            ),
        }
    else:
        report_urls = {}

    return AgentResponse(
        status=_workflow_status(result),
        request_id=request_id,
        route=result.get("route"),
        summary=_build_public_summary(
            result
        ),
        verification=_compact_verification(
            result
        ),
        rag_sources=_compact_rag_sources(
            result
        ),
        report_urls=report_urls,
        execution_trace=_json_safe(
            result.get(
                "execution_trace",
                [],
            )
        ),
        errors=_json_safe(
            result.get("errors", [])
        ),
    )


@app.post(
    "/api/v1/analyze",
    response_model=AgentResponse,
    tags=["analysis"],
    summary="Analyze one image",
)
async def analyze_image(
    image: UploadFile = File(...),
    query: str = Form(
        default=DEFAULT_SINGLE_QUERY
    ),
) -> AgentResponse:
    """Run the complete single-image IQA Agent workflow."""

    with tempfile.TemporaryDirectory(
        prefix="cot_iqa_analyze_"
    ) as temporary_directory:
        directory = Path(
            temporary_directory
        )

        image_path = await _save_upload(
            image,
            directory,
            "image",
        )

        initial_state = create_initial_state(
            user_query=(
                query.strip()
                or DEFAULT_SINGLE_QUERY
            ),
            image_paths=[
                str(image_path),
            ],
        )

        try:
            result = await _invoke_workflow(
                initial_state,
                recursion_limit=30,
            )

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            ) from exc

        return _build_agent_response(
            result
        )


@app.post(
    "/api/v1/compare",
    response_model=AgentResponse,
    tags=["analysis"],
    summary="Compare two images",
)
async def compare_images_api(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    query: str = Form(
        default=DEFAULT_COMPARISON_QUERY
    ),
) -> AgentResponse:
    """Run the complete two-image IQA Agent workflow."""

    with tempfile.TemporaryDirectory(
        prefix="cot_iqa_compare_"
    ) as temporary_directory:
        directory = Path(
            temporary_directory
        )

        image_1_path = await _save_upload(
            image_1,
            directory,
            "image_1",
        )

        image_2_path = await _save_upload(
            image_2,
            directory,
            "image_2",
        )

        initial_state = create_initial_state(
            user_query=(
                query.strip()
                or DEFAULT_COMPARISON_QUERY
            ),
            image_paths=[
                str(image_1_path),
                str(image_2_path),
            ],
        )

        try:
            result = await _invoke_workflow(
                initial_state,
                recursion_limit=40,
            )

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            ) from exc

        return _build_agent_response(
            result
        )



_REQUEST_ID_PATTERN = re.compile(
    r"^req_[A-Za-z0-9]+$"
)

_REPORT_TYPES = (
    "single_image",
    "comparison",
)


class ReportFileInfo(BaseModel):
    """Download information for one report file."""

    filename: str
    media_type: str
    size_bytes: int
    download_url: str


class ReportResponse(BaseModel):
    """Response containing one generated Agent report."""

    status: str
    request_id: str
    report_type: str
    json_report: dict[str, Any]
    markdown_report: str
    files: dict[str, ReportFileInfo]


def _get_report_directory() -> Path:
    """Resolve the configured report directory."""

    report_directory = _resolve_project_path(
        _get_config_value(
            "paths",
            "report_dir",
            default="outputs/reports",
        )
    )

    if report_directory is None:
        raise RuntimeError(
            "Report directory is not configured."
        )

    return report_directory


def _validate_request_id(
    request_id: str,
) -> str:
    """Reject unsafe or malformed report identifiers."""

    normalized = request_id.strip()

    if not _REQUEST_ID_PATTERN.fullmatch(
        normalized
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid request_id. Expected a value "
                "such as req_4d51bf443583."
            ),
        )

    return normalized


def _find_report_files(
    request_id: str,
) -> tuple[str, Path, Path]:
    """Locate the JSON and Markdown reports for a request."""

    normalized = _validate_request_id(
        request_id
    )

    report_directory = (
        _get_report_directory()
    )

    if not report_directory.is_dir():
        raise HTTPException(
            status_code=503,
            detail=(
                "Report directory is unavailable."
            ),
        )

    partial_matches: list[str] = []

    for report_type in _REPORT_TYPES:
        json_path = (
            report_directory
            / f"{normalized}_{report_type}.json"
        )

        markdown_path = (
            report_directory
            / f"{normalized}_{report_type}.md"
        )

        json_exists = json_path.is_file()
        markdown_exists = markdown_path.is_file()

        if json_exists and markdown_exists:
            return (
                report_type,
                json_path,
                markdown_path,
            )

        if json_exists or markdown_exists:
            partial_matches.append(
                report_type
            )

    if partial_matches:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "The report is incomplete."
                ),
                "request_id": normalized,
                "partial_report_types": (
                    partial_matches
                ),
            },
        )

    raise HTTPException(
        status_code=404,
        detail={
            "message": "Report not found.",
            "request_id": normalized,
        },
    )


@app.get(
    "/api/v1/reports/{request_id}",
    response_model=ReportResponse,
    tags=["reports"],
    summary="Read a generated report",
)
def get_report(
    request_id: str,
) -> ReportResponse:
    """Return JSON and Markdown reports for one request."""

    (
        report_type,
        json_path,
        markdown_path,
    ) = _find_report_files(request_id)

    try:
        json_report = json.loads(
            json_path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "The JSON report is invalid."
                ),
                "filename": json_path.name,
            },
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Failed to read the JSON report."
                ),
                "filename": json_path.name,
            },
        ) from exc

    try:
        markdown_report = (
            markdown_path.read_text(
                encoding="utf-8",
            )
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Failed to read the Markdown report."
                ),
                "filename": markdown_path.name,
            },
        ) from exc

    normalized = _validate_request_id(
        request_id
    )

    return ReportResponse(
        status="ok",
        request_id=normalized,
        report_type=report_type,
        json_report=_json_safe(
            json_report
        ),
        markdown_report=markdown_report,
        files={
            "json": ReportFileInfo(
                filename=json_path.name,
                media_type=(
                    "application/json"
                ),
                size_bytes=(
                    json_path.stat().st_size
                ),
                download_url=(
                    f"/api/v1/reports/"
                    f"{normalized}/download/json"
                ),
            ),
            "markdown": ReportFileInfo(
                filename=markdown_path.name,
                media_type=(
                    "text/markdown"
                ),
                size_bytes=(
                    markdown_path.stat().st_size
                ),
                download_url=(
                    f"/api/v1/reports/"
                    f"{normalized}/download/markdown"
                ),
            ),
        },
    )


@app.get(
    (
        "/api/v1/reports/{request_id}"
        "/download/{report_format}"
    ),
    tags=["reports"],
    summary="Download a generated report file",
)
def download_report(
    request_id: str,
    report_format: str,
) -> FileResponse:
    """Download a report as JSON or Markdown."""

    (
        _report_type,
        json_path,
        markdown_path,
    ) = _find_report_files(request_id)

    normalized_format = (
        report_format.strip().lower()
    )

    if normalized_format == "json":
        selected_path = json_path
        media_type = "application/json"

    elif normalized_format in {
        "md",
        "markdown",
    }:
        selected_path = markdown_path
        media_type = "text/markdown"

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported report format. "
                "Use json or markdown."
            ),
        )

    return FileResponse(
        path=selected_path,
        media_type=media_type,
        filename=selected_path.name,
    )



_ALLOWED_REPORT_TAGS = set(
    bleach.sanitizer.ALLOWED_TAGS
).union(
    {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "pre",
        "code",
        "hr",
        "br",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "details",
        "summary",
    }
)

_ALLOWED_REPORT_ATTRIBUTES = {
    "a": [
        "href",
        "title",
        "target",
        "rel",
    ],
    "code": [
        "class",
    ],
    "th": [
        "align",
    ],
    "td": [
        "align",
    ],
}


@app.get(
    "/api/v1/reports/{request_id}/view",
    response_class=HTMLResponse,
    tags=["reports"],
    summary="View a generated report as HTML",
)
def view_report(
    request_id: str,
) -> HTMLResponse:
    """Render a generated Markdown report as a readable webpage."""

    (
        report_type,
        _json_path,
        markdown_path,
    ) = _find_report_files(request_id)

    try:
        markdown_text = markdown_path.read_text(
            encoding="utf-8",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Failed to read the Markdown report."
                ),
                "filename": markdown_path.name,
            },
        ) from exc

    rendered_html = markdown_lib.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
        ],
        output_format="html5",
    )

    safe_report_html = bleach.clean(
        rendered_html,
        tags=_ALLOWED_REPORT_TAGS,
        attributes=_ALLOWED_REPORT_ATTRIBUTES,
        protocols=[
            "http",
            "https",
            "mailto",
        ],
        strip=True,
    )

    normalized = _validate_request_id(
        request_id
    )

    escaped_request_id = html.escape(
        normalized
    )

    escaped_report_type = html.escape(
        report_type
    )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1"
  >
  <title>{escaped_request_id} - COT-IQA-Agent Report</title>

  <style>
    :root {{
      color-scheme: light;
      font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        "Microsoft YaHei",
        Arial,
        sans-serif;
    }}

    body {{
      margin: 0;
      background: #f5f7fa;
      color: #1f2937;
      line-height: 1.7;
    }}

    .page {{
      max-width: 1080px;
      margin: 32px auto;
      padding: 0 20px 48px;
    }}

    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
      padding: 18px 22px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
    }}

    .toolbar h1 {{
      margin: 0 0 4px;
      font-size: 22px;
    }}

    .meta {{
      color: #6b7280;
      font-size: 14px;
    }}

    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .button {{
      display: inline-block;
      padding: 8px 14px;
      border-radius: 8px;
      background: #2563eb;
      color: #ffffff;
      text-decoration: none;
      font-size: 14px;
    }}

    .button.secondary {{
      background: #4b5563;
    }}

    .report {{
      padding: 28px 34px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      overflow-wrap: anywhere;
    }}

    .report h1 {{
      padding-bottom: 10px;
      border-bottom: 2px solid #e5e7eb;
    }}

    .report h2 {{
      margin-top: 32px;
      padding-bottom: 7px;
      border-bottom: 1px solid #e5e7eb;
    }}

    .report h3 {{
      margin-top: 25px;
    }}

    .report table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
    }}

    .report th,
    .report td {{
      padding: 9px 11px;
      border: 1px solid #d1d5db;
      text-align: left;
      vertical-align: top;
    }}

    .report th {{
      background: #f3f4f6;
    }}

    .report pre {{
      overflow-x: auto;
      padding: 16px;
      border-radius: 8px;
      background: #111827;
      color: #e5e7eb;
    }}

    .report code {{
      padding: 2px 5px;
      border-radius: 4px;
      background: #f3f4f6;
    }}

    .report pre code {{
      padding: 0;
      background: transparent;
    }}

    @media (max-width: 700px) {{
      .page {{
        margin-top: 16px;
        padding: 0 10px 30px;
      }}

      .report {{
        padding: 20px 17px;
      }}
    }}
  </style>
</head>

<body>
  <main class="page">
    <header class="toolbar">
      <div>
        <h1>COT-IQA-Agent 分析报告</h1>
        <div class="meta">
          请求编号：{escaped_request_id}
          · 类型：{escaped_report_type}
        </div>
      </div>

      <nav class="actions">
        <a
          class="button"
          href="/api/v1/reports/{escaped_request_id}/download/markdown"
        >
          下载 Markdown
        </a>

        <a
          class="button secondary"
          href="/api/v1/reports/{escaped_request_id}/download/json"
        >
          下载 JSON
        </a>
      </nav>
    </header>

    <article class="report">
      {safe_report_html}
    </article>
  </main>
</body>
</html>
"""

    return HTMLResponse(
        content=page,
        status_code=200,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
