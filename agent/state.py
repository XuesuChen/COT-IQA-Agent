"""Shared state definition for the COT-IQA Agent workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


TaskType = Literal[
    "single_image_analysis",
    "multi_image_comparison",
    "unknown",
]

RouteName = Literal[
    "single_image",
    "comparison",
    "reject",
]


class IQAAgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # ---------------------------------------------------------
    # Original request
    # ---------------------------------------------------------

    request_id: str
    user_query: str
    image_paths: list[str]
    reference_image_path: str | None

    # ---------------------------------------------------------
    # Conversation messages
    # ---------------------------------------------------------

    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    # ---------------------------------------------------------
    # Routing and planning
    # ---------------------------------------------------------

    task_type: TaskType
    route: RouteName
    plan: list[str]

    # ---------------------------------------------------------
    # Single-image preprocessing and inspection
    # ---------------------------------------------------------

    image_metadata: dict[str, Any]
    image_analysis: dict[str, Any]

    # ---------------------------------------------------------
    # Single-image IQA outputs
    # ---------------------------------------------------------

    pyiqa_results: dict[str, Any]
    cot_iqa_raw_output: str
    cot_iqa_result: dict[str, Any]

    # ---------------------------------------------------------
    # Comparison workflow
    # ---------------------------------------------------------

    comparison_items: list[dict[str, Any]]
    comparison_image_metadata: dict[str, Any]
    comparison_image_analysis: dict[str, Any]
    comparison_pyiqa_results: dict[str, Any]
    comparison_cot_results: dict[str, Any]
    comparison_result: dict[str, Any]

    # ---------------------------------------------------------
    # RAG and verification
    # ---------------------------------------------------------

    rag_context: list[dict[str, Any]]
    verification_result: dict[str, Any]

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    final_result: dict[str, Any]
    final_report: str

    # ---------------------------------------------------------
    # Append-only workflow records
    # ---------------------------------------------------------

    execution_trace: Annotated[
        list[str],
        operator.add,
    ]

    errors: Annotated[
        list[str],
        operator.add,
    ]


def create_initial_state(
    user_query: str,
    image_paths: list[str] | None = None,
    reference_image_path: str | None = None,
) -> IQAAgentState:
    """Create a clean initial state for one Agent request."""

    if not isinstance(user_query, str):
        raise TypeError(
            "user_query must be a string."
        )

    normalized_query = user_query.strip()

    if not normalized_query:
        raise ValueError(
            "user_query must not be empty."
        )

    normalized_paths = [
        str(path).strip()
        for path in (image_paths or [])
        if str(path).strip()
    ]

    normalized_reference = (
        reference_image_path.strip()
        if isinstance(
            reference_image_path,
            str,
        )
        and reference_image_path.strip()
        else None
    )

    return IQAAgentState(
        request_id=(
            f"req_{uuid4().hex[:12]}"
        ),
        user_query=normalized_query,
        image_paths=normalized_paths,
        reference_image_path=(
            normalized_reference
        ),
        messages=[],
        task_type="unknown",
        route="reject",
        plan=[],

        # Single-image state
        image_metadata={},
        image_analysis={},
        pyiqa_results={},
        cot_iqa_raw_output="",
        cot_iqa_result={},

        # Comparison state
        comparison_items=[],
        comparison_image_metadata={},
        comparison_image_analysis={},
        comparison_pyiqa_results={},
        comparison_cot_results={},
        comparison_result={},

        # Shared downstream state
        rag_context=[],
        verification_result={},
        final_result={},
        final_report="",
        execution_trace=[
            "state_initialized"
        ],
        errors=[],
    )
