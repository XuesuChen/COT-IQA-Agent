"""Deterministic request router for the COT-IQA Agent."""

from __future__ import annotations

from typing import Any

from agent.state import IQAAgentState, RouteName, TaskType


COMPARISON_KEYWORDS = {
    "对比",
    "比较",
    "哪张更好",
    "哪个好",
    "质量排序",
    "排名",
    "compare",
    "comparison",
    "better",
    "best",
    "rank",
}


SINGLE_IMAGE_PLAN = [
    "validate_image",
    "inspect_image",
    "run_pyiqa",
    "run_cot_iqa",
    "retrieve_knowledge",
    "verify_results",
    "generate_report",
]


COMPARISON_PLAN = [
    "validate_images",
    "inspect_images",
    "run_pyiqa_batch",
    "run_cot_iqa_batch",
    "compare_results",
    "retrieve_knowledge",
    "verify_results",
    "generate_report",
]


def _contains_comparison_intent(query: str) -> bool:
    """Check whether the user explicitly requests image comparison."""
    normalized_query = query.casefold()

    return any(
        keyword.casefold() in normalized_query
        for keyword in COMPARISON_KEYWORDS
    )


def route_request(state: IQAAgentState) -> dict[str, Any]:
    """Classify a request and return a partial state update."""

    user_query = state.get("user_query", "").strip()

    image_paths = [
        path.strip()
        for path in state.get("image_paths", [])
        if isinstance(path, str) and path.strip()
    ]

    reference_image_path = state.get("reference_image_path")

    has_reference = (
        isinstance(reference_image_path, str)
        and bool(reference_image_path.strip())
    )

    image_count = len(image_paths) + int(has_reference)
    comparison_requested = _contains_comparison_intent(user_query)

    task_type: TaskType
    route: RouteName
    plan: list[str]
    errors: list[str] = []

    if image_count == 0:
        task_type = "unknown"
        route = "reject"
        plan = []
        errors.append("No image was provided.")

    elif image_count == 1 and comparison_requested:
        task_type = "multi_image_comparison"
        route = "reject"
        plan = []
        errors.append(
            "A comparison request requires at least two images, "
            "or one target image and one reference image."
        )

    elif image_count == 1:
        task_type = "single_image_analysis"
        route = "single_image"
        plan = SINGLE_IMAGE_PLAN.copy()

    else:
        task_type = "multi_image_comparison"
        route = "comparison"
        plan = COMPARISON_PLAN.copy()

    update: dict[str, Any] = {
        "task_type": task_type,
        "route": route,
        "plan": plan,
        "execution_trace": [f"router:{route}"],
    }

    if errors:
        update["errors"] = errors

    return update


def select_route(state: IQAAgentState) -> RouteName:
    """Return the route used by a LangGraph conditional edge."""
    return state.get("route", "reject")
