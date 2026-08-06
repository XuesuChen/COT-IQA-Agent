"""LangGraph workflow definition for COT-IQA-Agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    compare_results,
    generate_comparison_report,
    generate_single_image_report,
    inspect_comparison_images,
    inspect_single_image,
    run_cot_iqa_comparison,
    run_cot_iqa_single_image,
    run_pyiqa_comparison,
    run_pyiqa_single_image,
    select_after_comparison_inspection,
    select_after_image_inspection,
    verify_comparison_results,
    verify_single_image_results,
)
from agent.nodes import retrieve_knowledge
from agent.router import route_request, select_route
from agent.state import IQAAgentState


def reject_request(
    state: IQAAgentState,
) -> dict[str, Any]:
    """Return a structured result for rejected requests."""

    request_id = state.get(
        "request_id",
        "unknown_request",
    )

    task_type = state.get(
        "task_type",
        "unknown",
    )

    route_errors = list(
        state.get("errors", [])
        or []
    )

    if route_errors:
        reason = "; ".join(route_errors)
    else:
        reason = "The request could not be routed."

    final_result = {
        "request_id": request_id,
        "task_type": task_type,
        "status": "rejected",
        "route": "reject",
        "reason": reason,
        "errors": route_errors,
        "report_json_path": None,
        "report_markdown_path": None,
    }

    final_report = "\n".join(
        [
            "# COT-IQA-Agent Request Rejected",
            "",
            f"- **Request ID:** `{request_id}`",
            f"- **Task type:** `{task_type}`",
            f"- **Reason:** {reason}",
            "",
        ]
    )

    return {
        "final_result": final_result,
        "final_report": final_report,
        "execution_trace": [
            "node:reject_request",
        ],
    }


def build_graph():
    """Build and compile the complete IQA Agent graph."""

    workflow = StateGraph(
        IQAAgentState
    )

    # ---------------------------------------------------------
    # Router
    # ---------------------------------------------------------

    workflow.add_node(
        "router",
        route_request,
    )

    workflow.add_node(
        "reject_request",
        reject_request,
    )

    # ---------------------------------------------------------
    # Single-image workflow
    # ---------------------------------------------------------

    workflow.add_node(
        "inspect_single_image",
        inspect_single_image,
    )

    workflow.add_node(
        "run_pyiqa_single_image",
        run_pyiqa_single_image,
    )

    workflow.add_node(
        "run_cot_iqa_single_image",
        run_cot_iqa_single_image,
    )

    workflow.add_node(
        "verify_single_image_results",
        verify_single_image_results,
    )

    workflow.add_node(
        "generate_single_image_report",
        generate_single_image_report,
    )

    # ---------------------------------------------------------
    # Comparison workflow
    # ---------------------------------------------------------

    workflow.add_node(
        "inspect_comparison_images",
        inspect_comparison_images,
    )

    workflow.add_node(
        "run_pyiqa_comparison",
        run_pyiqa_comparison,
    )

    workflow.add_node(
        "run_cot_iqa_comparison",
        run_cot_iqa_comparison,
    )

    workflow.add_node(
        "compare_results",
        compare_results,
    )

    workflow.add_node(
        "verify_comparison_results",
        verify_comparison_results,
    )

    workflow.add_node(
        "generate_comparison_report",
        generate_comparison_report,
    )

    # ---------------------------------------------------------
    # Entry routing
    # ---------------------------------------------------------

    workflow.add_node(
        "retrieve_single_image_knowledge",
        retrieve_knowledge,
    )
    workflow.add_node(
        "retrieve_comparison_knowledge",
        retrieve_knowledge,
    )


    workflow.add_edge(
        START,
        "router",
    )

    workflow.add_conditional_edges(
        "router",
        select_route,
        {
            "single_image": (
                "inspect_single_image"
            ),
            "comparison": (
                "inspect_comparison_images"
            ),
            "reject": (
                "reject_request"
            ),
        },
    )

    # ---------------------------------------------------------
    # Single-image edges
    # ---------------------------------------------------------

    workflow.add_conditional_edges(
        "inspect_single_image",
        select_after_image_inspection,
        {
            "valid": (
                "run_pyiqa_single_image"
            ),
            "invalid": (
                "verify_single_image_results"
            ),
        },
    )

    workflow.add_edge(
        "run_pyiqa_single_image",
        "run_cot_iqa_single_image",
    )

    workflow.add_edge(
        "run_cot_iqa_single_image",
        "retrieve_single_image_knowledge",
    )
    workflow.add_edge(
        "retrieve_single_image_knowledge",
        "verify_single_image_results",
    )

    workflow.add_edge(
        "verify_single_image_results",
        "generate_single_image_report",
    )

    workflow.add_edge(
        "generate_single_image_report",
        END,
    )

    # ---------------------------------------------------------
    # Comparison edges
    # ---------------------------------------------------------

    workflow.add_conditional_edges(
        "inspect_comparison_images",
        select_after_comparison_inspection,
        {
            "valid": (
                "run_pyiqa_comparison"
            ),
            "invalid": (
                "verify_comparison_results"
            ),
        },
    )

    workflow.add_edge(
        "run_pyiqa_comparison",
        "run_cot_iqa_comparison",
    )

    workflow.add_edge(
        "run_cot_iqa_comparison",
        "compare_results",
    )

    workflow.add_edge(
        "compare_results",
        "retrieve_comparison_knowledge",
    )
    workflow.add_edge(
        "retrieve_comparison_knowledge",
        "verify_comparison_results",
    )

    workflow.add_edge(
        "verify_comparison_results",
        "generate_comparison_report",
    )

    workflow.add_edge(
        "generate_comparison_report",
        END,
    )

    # ---------------------------------------------------------
    # Rejection edge
    # ---------------------------------------------------------

    workflow.add_edge(
        "reject_request",
        END,
    )

    return workflow.compile()


graph = build_graph()

# Compatibility aliases for later API/UI integration.
compiled_graph = graph
app = graph
