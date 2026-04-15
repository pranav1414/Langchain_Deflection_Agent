from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import (
    classify_intent,
    retrieve_context,
    generate_answer,
    collect_feedback,
    resolve_or_escalate
)


def should_continue_after_retrieval(state: AgentState) -> str:
    """
    Edge decision after Node 2.
    If Tier 3 was detected, skip to end.
    Otherwise continue to answer generation.
    """
    if state.get("outcome") == "escalated":
        return "end"
    return "generate"


def should_continue_after_resolution(state: AgentState) -> str:
    """
    Edge decision after Node 5.
    If retry_count was reset (user said No, first attempt),
    loop back to answer generation with new docs.
    If deflected or escalated, go to end.
    """
    outcome = state.get("outcome")
    feedback = state.get("feedback")
    retry_count = state.get("retry_count", 0)

    if outcome == "deflected":
        return "end"
    
    if outcome == "escalated":
        return "end"
    
    # feedback is None means retry was triggered
    if feedback is None and retry_count > 0:
        return "retry"
    
    return "end"


def build_graph() -> StateGraph:
    """
    Builds and compiles the full LangGraph agent.
    
    Flow:
    classify → retrieve → [tier 3 check] → generate → feedback → resolve
                                                              ↑         |
                                                              └─────────┘
                                                           (retry loop)
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", generate_answer)
    graph.add_node("feedback", collect_feedback)
    graph.add_node("resolve", resolve_or_escalate)

    # Set entry point
    graph.set_entry_point("classify")

    # Fixed edges
    graph.add_edge("classify", "retrieve")
    graph.add_edge("generate", "feedback")
    graph.add_edge("feedback", "resolve")

    # Conditional edge after retrieval
    graph.add_conditional_edges(
        "retrieve",
        should_continue_after_retrieval,
        {
            "generate": "generate",
            "end": END
        }
    )

    # Conditional edge after resolution
    graph.add_conditional_edges(
        "resolve",
        should_continue_after_resolution,
        {
            "end": END,
            "retry": "generate"
        }
    )

    return graph.compile()


# Compile once at import time
agent = build_graph()