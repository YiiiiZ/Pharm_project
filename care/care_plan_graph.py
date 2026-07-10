"""LangGraph assembly for the care-plan generation workflow.

Build and compile the graph with build_care_plan_graph(). The returned
compiled graph can be invoked with:

    compiled = build_care_plan_graph(provider="anthropic", evaluator=my_evaluator)
    result = compiled.invoke(initial_state)

Call save_graph_diagram() to write the Mermaid diagram to a file.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from .generation_loop import Evaluator, HallucinationReport
from .graph_edges import route_after_format, route_after_verify
from .graph_nodes import (
    auto_fix_node,
    format_check_node,
    make_generate_node,
    make_verify_node,
    route_node,
)
from .graph_state import CarePlanState


def build_care_plan_graph(
    *,
    provider: str = "anthropic",
    client: Any | None = None,
    evaluator: Evaluator | None = None,
):
    """Compile and return the care-plan StateGraph.

    Args:
        provider:  "anthropic" or "openai"
        client:    Pre-built LLM client (optional; created from env vars if absent)
        evaluator: Hallucination evaluator; defaults to a pass-through that
                   always returns valid so the graph can be tested without an LLM

    Graph shape:
        route ──► generate ──► format_check ──[route_after_format]──► verify
                    ▲                │ (exhausted)                        │
                    │                └──────────────────────────────► END │
                    │                                                      │
                    ├──────────────────────── [reasoning / retries left] ◄─┤
                    │                                                      │
                   END ◄──────────────── [success / escalated / exhausted]─┤
                                                                           │
                                    auto_fix ──► END ◄── [all numerical] ──┘
    """
    _evaluator = evaluator or _passthrough_evaluator

    builder = StateGraph(CarePlanState)

    # ---- nodes ----
    builder.add_node("route", route_node)
    builder.add_node("generate", make_generate_node(provider=provider, client=client))
    builder.add_node("format_check", format_check_node)
    builder.add_node("verify", make_verify_node(_evaluator))
    builder.add_node("auto_fix", auto_fix_node)

    # ---- unconditional edges ----
    builder.set_entry_point("route")
    builder.add_edge("route", "generate")
    builder.add_edge("generate", "format_check")
    builder.add_edge("auto_fix", END)

    # ---- conditional edge: after format_check ----
    builder.add_conditional_edges(
        "format_check",
        route_after_format,
        {
            "verify": "verify",
            "generate": "generate",
            "end": END,
        },
    )

    # ---- conditional edge: after verify ----
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "end": END,
            "auto_fix": "auto_fix",
            "generate": "generate",
        },
    )

    return builder.compile()


def save_graph_diagram(path: str = "care_plan_graph.md") -> str:
    """Compile the graph, render its Mermaid diagram, and write it to path.

    Returns the Mermaid string so callers can also print or embed it.
    """
    compiled = build_care_plan_graph()
    mermaid = compiled.get_graph().draw_mermaid()
    with open(path, "w", encoding="utf-8") as f:
        f.write("```mermaid\n")
        f.write(mermaid)
        f.write("```\n")
    return mermaid


def make_initial_state(
    *,
    order_id: int,
    patient_data: dict[str, Any],
    initial_prompt: str,
) -> CarePlanState:
    """Return a fully initialised CarePlanState ready to feed into the graph."""
    return CarePlanState(
        order_id=order_id,
        patient_data=patient_data,
        initial_prompt=initial_prompt,
        routing_strategy=None,
        current_prompt_version="v1",
        current_model="",
        llm_output=None,
        parsed_output=None,
        verification_report=None,
        generate_retry_count=0,
        format_retry_count=0,
        repair_retry_count=0,
        error_log=[],
        accumulated_feedback="",
        status="pending",
        escalate_to_pharmacist=False,
        final_output=None,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _passthrough_evaluator(output: str) -> HallucinationReport:
    """Default evaluator for testing: always returns valid with no findings."""
    return HallucinationReport(valid=True, hallucination_count=0)
