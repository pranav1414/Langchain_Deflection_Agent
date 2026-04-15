import os
import sys
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "LangChain Deflection Agent")

import streamlit as st
from src.nodes import classify_intent, retrieve_context, generate_answer, resolve_or_escalate
from src.tools import get_collection_stats


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LangChain Support Deflection Agent",
    page_icon="assets/langchain_logo.jpeg",
    layout="wide"
)


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    color: #1a1a2e !important;
    margin-bottom: 0.2rem !important;
    line-height: 1.2 !important;
}
.sub-header {
    font-size: 1.5rem !important;
    color: #555 !important;
    margin-bottom: 1.5rem !important;
}
    .tier-badge-1 {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .tier-badge-2 {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .tier-badge-3 {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .outcome-deflected {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
    }
    .outcome-escalated {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
    }
    .answer-box {
        background-color: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 16px;
        border-radius: 4px;
        margin: 12px 0;
    }
    .escalation-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 16px;
        border-radius: 4px;
        margin: 12px 0;
    }
    .history-item {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 3px solid #dee2e6;
    }
    .history-deflected {
        border-left: 3px solid #28a745;
    }
    .history-escalated {
        border-left: 3px solid #dc3545;
    }
    .trace-link {
        background-color: #e8f4f8;
        border: 1px solid #0066cc;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.85rem;
    }
    .langsmith-badge {
        background-color: #0066cc;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "deflection_count" not in st.session_state:
    st.session_state.deflection_count = 0
if "escalation_count" not in st.session_state:
    st.session_state.escalation_count = 0
if "total_count" not in st.session_state:
    st.session_state.total_count = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_tier_label(tier: int) -> str:
    labels = {
        1: "Tier 1 — Simple How-To",
        2: "Tier 2 — Debugging",
        3: "Tier 3 — Architecture"
    }
    return labels.get(tier, "Unknown")


def get_tier_color(tier: int) -> str:
    colors = {1: "green", 2: "orange", 3: "red"}
    return colors.get(tier, "gray")


def get_langsmith_url() -> str:
    return "https://smith.langchain.com"


def run_classification_and_retrieval(question: str) -> dict:
    initial_state = {
        "question": question,
        "tier": None,
        "classification_confidence": None,
        "classification_reasoning": None,
        "retrieved_docs": None,
        "retrieval_sources": None,
        "answer": None,
        "answer_confidence": None,
        "citations": None,
        "feedback": None,
        "retry_count": 0,
        "outcome": None,
        "escalation_context": None,
        "error": None
    }

    state = classify_intent(initial_state)

    if state.get("tier") == 3:
        state["outcome"] = "escalated"
        state["escalation_context"] = {
            "reason": "Tier 3 question requires human investigation",
            "question": question,
            "tier": 3
        }
        return state

    state = retrieve_context(state)
    state = generate_answer(state)
    return state


def run_resolution(state: dict, feedback: str) -> dict:
    state["feedback"] = feedback
    state = resolve_or_escalate(state)
    return state


def run_retry(state: dict) -> dict:
    state = retrieve_context(state)
    state = generate_answer(state)
    return state


def add_to_history(state: dict):
    st.session_state.history.append({
        "question": state.get("question", ""),
        "tier": state.get("tier"),
        "outcome": state.get("outcome"),
        "confidence": state.get("answer_confidence"),
        "retries": state.get("retry_count", 0),
        "langsmith_url": get_langsmith_url()
    })


# ─────────────────────────────────────────────
# HEADER — Large and prominent
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("assets/langchain_logo.jpeg", width=60)
with col_title:
    st.markdown('<p class="main-header">LangChain Support Deflection Agent</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI powered support that resolves issues before a human needs to step in</p>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Questions", st.session_state.total_count)
with col2:
    st.metric("✅ Deflected", st.session_state.deflection_count)
with col3:
    st.metric("🎫 Escalated", st.session_state.escalation_count)
with col4:
    deflection_rate = (
        st.session_state.deflection_count /
        st.session_state.total_count * 100
        if st.session_state.total_count > 0 else 0
    )
    st.metric("Deflection Rate", f"{deflection_rate:.0f}%")

st.divider()

# ─────────────────────────────────────────────
# DB STATUS CHECK
# ─────────────────────────────────────────────
stats = get_collection_stats()
if stats["docs_count"] == 0:
    st.error("⚠️ ChromaDB is empty. Run: python data/ingest.py")
    st.stop()

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
left_col, right_col = st.columns([2, 1])

with left_col:

    # ─────────────────────────────────────────
    # STAGE 1 - Input
    # ─────────────────────────────────────────
    if st.session_state.stage == "input":
        st.subheader("Ask a LangChain Question")

        st.markdown("**Try these examples:**")
        ex_col1, ex_col2, ex_col3 = st.columns(3)

        with ex_col1:
            if st.button("How do I add a checkpointer?",
                         use_container_width=True):
                st.session_state.example_question = (
                    "How do I add a checkpointer to my LangGraph agent?"
                )
        with ex_col2:
            if st.button("Streaming broken after upgrade",
                         use_container_width=True):
                st.session_state.example_question = (
                    "Why is my token streaming not working "
                    "after upgrading LangGraph?"
                )
        with ex_col3:
            if st.button("Multi-tenant architecture",
                         use_container_width=True):
                st.session_state.example_question = (
                    "How do I architect a multi-tenant "
                    "LangGraph deployment at 10k RPS?"
                )

        default_q = st.session_state.get("example_question", "")
        question = st.text_area(
            "Your question:",
            value=default_q,
            height=100,
            placeholder="Ask anything about LangChain or LangGraph..."
        )

        if st.button("🔍 Find Answer", type="primary",
                     use_container_width=True):
            if question.strip():
                with st.spinner("Classifying and retrieving..."):
                    state = run_classification_and_retrieval(
                        question.strip()
                    )
                    st.session_state.agent_state = state

                    if state.get("tier") == 3:
                        st.session_state.stage = "tier3_escalation"
                    else:
                        st.session_state.stage = "feedback"
                st.rerun()
            else:
                st.warning("Please enter a question first.")

        # ─────────────────────────────────────
        # SESSION HISTORY
        # ─────────────────────────────────────
        if st.session_state.history:
            st.divider()
            st.subheader(
                f"Session History "
                f"({len(st.session_state.history)} questions)"
            )

            for i, item in enumerate(
                reversed(st.session_state.history)
            ):
                outcome = item.get("outcome", "unknown")
                tier = item.get("tier", "?")
                confidence = item.get("confidence", 0) or 0
                retries = item.get("retries", 0)

                border_class = (
                    "history-deflected"
                    if outcome == "deflected"
                    else "history-escalated"
                )
                outcome_icon = (
                    "✅" if outcome == "deflected" else "🎫"
                )
                tier_colors = {1: "🟢", 2: "🟡", 3: "🔴"}
                tier_icon = tier_colors.get(tier, "⚪")

                st.markdown(
                    f'<div class="history-item {border_class}">'
                    f'<strong>{outcome_icon} '
                    f'{item["question"][:70]}'
                    f'{"..." if len(item["question"]) > 70 else ""}'
                    f'</strong><br>'
                    f'<small>{tier_icon} Tier {tier} &nbsp;|&nbsp; '
                    f'Outcome: {outcome} &nbsp;|&nbsp; '
                    f'Confidence: {confidence:.0%} &nbsp;|&nbsp; '
                    f'Attempts: {retries + 1}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # LangSmith link for full history
            langsmith_url = get_langsmith_url()
            st.markdown(
                f'<div class="trace-link">'
                f'<span class="langsmith-badge">LangSmith</span> '
                f'&nbsp; View all traces → '
                f'<a href="{langsmith_url}" target="_blank">'
                f'smith.langchain.com</a>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ─────────────────────────────────────────
    # STAGE 2 - Answer + Feedback
    # ─────────────────────────────────────────
    elif st.session_state.stage == "feedback":
        state = st.session_state.agent_state
        retry_count = state.get("retry_count", 0)

        st.subheader("Here is what I found")

        answer = state.get("answer", "No answer generated")
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True
        )

        citations = state.get("citations", [])
        if citations:
            st.caption(f"Sources: {', '.join(citations)}")

        confidence = state.get("answer_confidence", 0)
        st.progress(
            float(confidence) if confidence else 0,
            text=f"Confidence: {confidence:.0%}"
            if confidence else "Confidence: Unknown"
        )

        # LangSmith trace link
        langsmith_url = get_langsmith_url()
        st.markdown(
            f'<div class="trace-link">'
            f'<span class="langsmith-badge">LangSmith</span>'
            f'&nbsp; View this trace → '
            f'<a href="{langsmith_url}" target="_blank">'
            f'smith.langchain.com</a>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.divider()

        if retry_count == 0:
            st.markdown("**Did this solve your problem?**")
        else:
            st.markdown("**Did this deeper search help?**")

        fb_col1, fb_col2 = st.columns(2)

        with fb_col1:
            if st.button("✅ Yes, solved!", type="primary",
                         use_container_width=True):
                state = run_resolution(state, "yes")
                st.session_state.agent_state = state
                st.session_state.stage = "deflected"
                st.session_state.deflection_count += 1
                st.session_state.total_count += 1
                add_to_history(state)
                st.rerun()

        with fb_col2:
            if st.button("❌ No, still stuck",
                         use_container_width=True):
                if retry_count < 1:
                    with st.spinner("Trying deeper search..."):
                        state["retry_count"] = 1
                        state["feedback"] = None
                        state = run_retry(state)
                        st.session_state.agent_state = state
                    st.rerun()
                else:
                    state = run_resolution(state, "no")
                    st.session_state.agent_state = state
                    st.session_state.stage = "escalated"
                    st.session_state.escalation_count += 1
                    st.session_state.total_count += 1
                    add_to_history(state)
                    st.rerun()

        if st.button("← Ask another question"):
            st.session_state.stage = "input"
            st.session_state.agent_state = None
            if "example_question" in st.session_state:
                del st.session_state.example_question
            st.rerun()

    # ─────────────────────────────────────────
    # STAGE 3A - Deflected
    # ─────────────────────────────────────────
    elif st.session_state.stage == "deflected":
        st.success("✅ Case Deflected Successfully")
        st.markdown(
            "Your question was resolved without needing "
            "a support ticket."
        )

        state = st.session_state.agent_state
        st.markdown(
            f'<div class="answer-box">'
            f'{state.get("answer", "")}'
            f'</div>',
            unsafe_allow_html=True
        )

        # LangSmith trace link
        langsmith_url = get_langsmith_url()
        st.markdown(
            f'<div class="trace-link">'
            f'<span class="langsmith-badge">LangSmith</span>'
            f'&nbsp; View trace → '
            f'<a href="{langsmith_url}" target="_blank">'
            f'smith.langchain.com</a>'
            f'</div>',
            unsafe_allow_html=True
        )

        if st.button("← Ask another question",
                     type="primary", use_container_width=True):
            st.session_state.stage = "input"
            st.session_state.agent_state = None
            if "example_question" in st.session_state:
                del st.session_state.example_question
            st.rerun()

    # ─────────────────────────────────────────
    # STAGE 3B - Escalated
    # ─────────────────────────────────────────
    elif st.session_state.stage == "escalated":
        st.warning("🎫 Escalated to Support Engineer")
        st.markdown(
            "A support engineer will follow up "
            "with full context."
        )

        state = st.session_state.agent_state
        ctx = state.get("escalation_context", {})

        if ctx:
            st.markdown(
                '<div class="escalation-box">',
                unsafe_allow_html=True
            )
            st.markdown("**Engineer Handoff Summary:**")
            st.markdown(
                ctx.get("engineer_summary",
                        "No summary available")
            )
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Full escalation context"):
                st.json({
                    "question": ctx.get("original_question"),
                    "tier": ctx.get("tier"),
                    "attempts": ctx.get("answers_attempted"),
                    "sources_checked": ctx.get("sources_checked"),
                    "confidence_at_escalation": ctx.get(
                        "confidence_at_escalation"
                    )
                })

        # LangSmith trace link
        langsmith_url = get_langsmith_url()
        st.markdown(
            f'<div class="trace-link">'
            f'<span class="langsmith-badge">LangSmith</span>'
            f'&nbsp; View escalation trace → '
            f'<a href="{langsmith_url}" target="_blank">'
            f'smith.langchain.com</a>'
            f'</div>',
            unsafe_allow_html=True
        )

        if st.button("← Ask another question",
                     type="primary", use_container_width=True):
            st.session_state.stage = "input"
            st.session_state.agent_state = None
            if "example_question" in st.session_state:
                del st.session_state.example_question
            st.rerun()

    # ─────────────────────────────────────────
    # STAGE 3C - Tier 3 Escalation
    # ─────────────────────────────────────────
    elif st.session_state.stage == "tier3_escalation":
        st.warning("🎫 Escalated — Architecture Question Detected")
        st.markdown("""
        This question requires deep production architecture
        expertise and codebase-level investigation.
        A senior engineer will assist you.
        """)

        state = st.session_state.agent_state
        st.info(f"**Question:** {state.get('question')}")
        st.caption(
            "Tier 3 questions are routed directly to engineers "
            "without retrieval — no wasted LLM calls."
        )

        # LangSmith trace link
        langsmith_url = get_langsmith_url()
        st.markdown(
            f'<div class="trace-link">'
            f'<span class="langsmith-badge">LangSmith</span>'
            f'&nbsp; View trace → '
            f'<a href="{langsmith_url}" target="_blank">'
            f'smith.langchain.com</a>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.session_state.escalation_count += 1
        st.session_state.total_count += 1
        add_to_history(state)

        if st.button("← Ask another question",
                     type="primary", use_container_width=True):
            st.session_state.stage = "input"
            st.session_state.agent_state = None
            if "example_question" in st.session_state:
                del st.session_state.example_question
            st.rerun()

# ─────────────────────────────────────────────
# RIGHT SIDEBAR - Agent Internals
# ─────────────────────────────────────────────
with right_col:
    st.subheader("Agent Internals")

    state = st.session_state.agent_state

    if state:
        tier = state.get("tier")
        if tier:
            st.markdown("**Classification:**")
            st.markdown(
                f'<span class="tier-badge-{tier}">'
                f'{get_tier_label(tier)}</span>',
                unsafe_allow_html=True
            )
            conf = state.get("classification_confidence", 0)
            st.caption(f"Confidence: {conf:.0%}" if conf else "")
            reasoning = state.get("classification_reasoning", "")
            if reasoning:
                st.caption(f"Reasoning: {reasoning}")

        st.divider()

        sources = state.get("retrieval_sources", [])
        if sources:
            st.markdown("**Sources Searched:**")
            for source in sources:
                st.markdown(f"- `{source}`")

        st.divider()

        outcome = state.get("outcome")
        if outcome:
            st.markdown("**Outcome:**")
            if outcome == "deflected":
                st.markdown(
                    '<span class="outcome-deflected">'
                    '✅ Deflected</span>',
                    unsafe_allow_html=True
                )
            elif outcome == "escalated":
                st.markdown(
                    '<span class="outcome-escalated">'
                    '🎫 Escalated</span>',
                    unsafe_allow_html=True
                )

        st.divider()

        retry = state.get("retry_count", 0)
        st.markdown(f"**Attempts:** {retry + 1}")

        confidence = state.get("answer_confidence")
        if confidence is not None:
            if confidence < 0.6:
                st.warning(
                    f"⚠️ Low confidence ({confidence:.0%}) — "
                    "consider contacting support directly"
                )
            else:
                st.caption(
                    f"Answer confidence: {confidence:.0%}"
                )

        error = state.get("error")
        if error:
            st.error(f"Error: {error}")

    else:
        st.info(
            "Agent internals will appear here "
            "once you ask a question."
        )

    st.divider()

    st.markdown("**Knowledge Base:**")
    st.caption(f"Docs: {stats['docs_count']} chunks")
    st.caption(f"Support KB: {stats['support_kb_count']} articles")

    st.divider()

    # LangSmith project link in sidebar
    langsmith_url = get_langsmith_url()
    st.markdown("**Observability:**")
    st.markdown(
        f'<span class="langsmith-badge">LangSmith</span> '
        f'<a href="{langsmith_url}" target="_blank">'
        f'View all traces</a>',
        unsafe_allow_html=True
    )