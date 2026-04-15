import json
import os
import logging
import time
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from src.state import AgentState
from src.tools import search_docs, search_support_kb, combined_search
from src.prompts import (
    CLASSIFICATION_PROMPT,
    ANSWER_GENERATION_PROMPT,
    RETRY_PROMPT,
    ESCALATION_SUMMARY_PROMPT
)


# ─────────────────────────────────────────────
# STRUCTURED JSON LOGGING
# ─────────────────────────────────────────────
logger = logging.getLogger("deflection_agent")
logger.setLevel(logging.INFO)

log_handler = logging.FileHandler("agent.log")
log_handler.setLevel(logging.INFO)
logger.addHandler(log_handler)


def log_node_event(
    node_name: str,
    question: str,
    details: dict,
    duration_ms: float = 0,
    error: str = None
):
    """
    Writes a structured JSON log entry for every node execution.
    In production these logs feed into monitoring tools like
    Datadog, CloudWatch, or Grafana for alerting and analysis.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "node": node_name,
        "question": question[:100],
        "duration_ms": round(duration_ms, 2),
        "error": error,
        **details
    }
    logger.info(json.dumps(log_entry))


# ─────────────────────────────────────────────
# INITIALIZE GEMINI
# ─────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)


# ─────────────────────────────────────────────
# NODE 1 — Intent Classification
# ─────────────────────────────────────────────
@traceable(name="Node1_IntentClassification")
def classify_intent(state: AgentState) -> AgentState:
    """
    Reads the question and classifies it into Tier 1, 2, or 3.
    Tier 1 = simple how-to
    Tier 2 = debugging/integration
    Tier 3 = deep production/architecture
    """
    start_time = time.time()
    try:
        prompt = CLASSIFICATION_PROMPT.format(question=state["question"])
        response = llm.invoke(prompt)

        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        duration = (time.time() - start_time) * 1000
        log_node_event(
            node_name="Node1_IntentClassification",
            question=state["question"],
            details={
                "tier": result["tier"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"]
            },
            duration_ms=duration
        )

        return {
            **state,
            "tier": result["tier"],
            "classification_confidence": result["confidence"],
            "classification_reasoning": result["reasoning"],
            "error": None
        }

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_node_event(
            node_name="Node1_IntentClassification",
            question=state["question"],
            details={"tier": 2, "confidence": 0.0},
            duration_ms=duration,
            error=str(e)
        )
        return {
            **state,
            "tier": 2,
            "classification_confidence": 0.0,
            "classification_reasoning": "Classification failed - defaulting to Tier 2",
            "error": f"Classification error: {str(e)}"
        }


# ─────────────────────────────────────────────
# NODE 2 — Tiered Retrieval
# ─────────────────────────────────────────────
@traceable(name="Node2_TieredRetrieval")
def retrieve_context(state: AgentState) -> AgentState:
    """
    Routes to the right retrieval source based on tier.
    Tier 1 = docs only
    Tier 2 = docs + support KB combined
    Tier 3 = flag for escalation, no retrieval needed
    """
    tier = state.get("tier", 2)
    question = state["question"]
    start_time = time.time()

    try:
        if tier == 1:
            results = search_docs(question, n_results=3)
            sources_label = ["langchain_docs"]

        elif tier == 2:
            results = combined_search(question, n_results=3)
            sources_label = ["langchain_docs", "support_kb"]

        else:
            # Tier 3 - too complex, flag for escalation
            # In production this would:
            # 1. POST escalation_context to ticketing system API (Zendesk/Linear)
            # 2. Fire Slack alert to #support-escalations channel
            # 3. Update CRM with interaction record
            # Current implementation logs to LangSmith and returns context
            duration = (time.time() - start_time) * 1000
            log_node_event(
                node_name="Node2_TieredRetrieval",
                question=question,
                details={
                    "tier": tier,
                    "action": "tier3_immediate_escalation",
                    "docs_retrieved": 0
                },
                duration_ms=duration
            )
            return {
                **state,
                "retrieved_docs": [],
                "retrieval_sources": ["none - tier 3 escalation"],
                "outcome": "escalated",
                "escalation_context": {
                    "reason": "Tier 3 question requires codebase level investigation",
                    "question": question,
                    "tier": tier
                },
                "error": None
            }

        # Handle empty retrieval - fallback
        if not results["found"]:
            duration = (time.time() - start_time) * 1000
            log_node_event(
                node_name="Node2_TieredRetrieval",
                question=question,
                details={
                    "tier": tier,
                    "action": "empty_retrieval_fallback",
                    "docs_retrieved": 0
                },
                duration_ms=duration,
                error="No documents found"
            )
            return {
                **state,
                "retrieved_docs": ["No relevant documentation found for this query."],
                "retrieval_sources": sources_label,
                "error": "No documents found - will attempt direct LLM answer"
            }

        duration = (time.time() - start_time) * 1000
        log_node_event(
            node_name="Node2_TieredRetrieval",
            question=question,
            details={
                "tier": tier,
                "sources": sources_label,
                "docs_retrieved": len(results["documents"])
            },
            duration_ms=duration
        )
        return {
            **state,
            "retrieved_docs": results["documents"],
            "retrieval_sources": sources_label,
            "error": None
        }

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_node_event(
            node_name="Node2_TieredRetrieval",
            question=question,
            details={"tier": tier, "docs_retrieved": 0},
            duration_ms=duration,
            error=str(e)
        )
        return {
            **state,
            "retrieved_docs": ["Retrieval failed - attempting direct answer"],
            "retrieval_sources": ["none"],
            "error": f"Retrieval error: {str(e)}"
        }


# ─────────────────────────────────────────────
# NODE 3 — Answer Generation
# ─────────────────────────────────────────────
@traceable(name="Node3_AnswerGeneration")
def generate_answer(state: AgentState) -> AgentState:
    """
    Takes retrieved docs and generates a structured answer.
    Includes confidence score and citations.
    Retries once if LLM call fails.
    """
    question = state["question"]
    docs = state.get("retrieved_docs", [])
    context = "\n\n---\n\n".join(docs) if docs else "No context available."
    retry_count = state.get("retry_count", 0)
    start_time = time.time()

    try:
        if retry_count > 0 and state.get("answer"):
            prompt = RETRY_PROMPT.format(
                question=question,
                previous_answer=state["answer"],
                context=context
            )
        else:
            prompt = ANSWER_GENERATION_PROMPT.format(
                question=question,
                context=context
            )

        response = llm.invoke(prompt)
        content = response.content.strip()

        answer = ""
        citations = []
        confidence = 0.5

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("ANSWER:"):
                answer_parts = [line.replace("ANSWER:", "").strip()]
                for next_line in lines[i+1:]:
                    if next_line.startswith("CITATIONS:") or next_line.startswith("CONFIDENCE:"):
                        break
                    answer_parts.append(next_line)
                answer = "\n".join(answer_parts).strip()

            elif line.startswith("CITATIONS:"):
                citations = [line.replace("CITATIONS:", "").strip()]

            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    conf_str = conf_str.split("/")[0].replace("%", "")
                    confidence = float(conf_str.strip())
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                except Exception:
                    confidence = 0.7

        if not answer:
            answer = content
            confidence = 0.5

        duration = (time.time() - start_time) * 1000
        log_node_event(
            node_name="Node3_AnswerGeneration",
            question=question,
            details={
                "confidence": confidence,
                "citations": citations,
                "retry_count": retry_count,
                "answer_length": len(answer)
            },
            duration_ms=duration
        )

        return {
            **state,
            "answer": answer,
            "answer_confidence": confidence,
            "citations": citations,
            "error": None
        }

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_node_event(
            node_name="Node3_AnswerGeneration",
            question=question,
            details={"confidence": 0.0, "retry_count": retry_count},
            duration_ms=duration,
            error=str(e)
        )
        if retry_count < 1:
            return {
                **state,
                "retry_count": retry_count + 1,
                "error": f"Answer generation failed, retrying: {str(e)}"
            }

        return {
            **state,
            "answer": "I was unable to generate an answer at this time. Please try again or contact support.",
            "answer_confidence": 0.0,
            "citations": [],
            "error": f"Answer generation failed after retry: {str(e)}"
        }


# ─────────────────────────────────────────────
# NODE 4 — Feedback Collection
# ─────────────────────────────────────────────
@traceable(name="Node4_FeedbackCollection")
def collect_feedback(state: AgentState) -> AgentState:
    """
    Presents the answer to the user and asks if it helped.
    This is the core of Finding 1 - the missing feedback loop.
    Captures yes/no and routes accordingly.
    """
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(state.get("answer", "No answer generated"))
    print("\nSources:", state.get("retrieval_sources", []))
    print("Confidence:", state.get("answer_confidence", 0))
    print("="*60)

    while True:
        feedback = input("\nDid this solve your problem? (yes/no): ").strip().lower()
        if feedback in ["yes", "no", "y", "n"]:
            break
        print("Please type 'yes' or 'no'")

    normalized = "yes" if feedback in ["yes", "y"] else "no"

    log_node_event(
        node_name="Node4_FeedbackCollection",
        question=state["question"],
        details={
            "feedback": normalized,
            "retry_count": state.get("retry_count", 0)
        }
    )

    return {
        **state,
        "feedback": normalized,
        "error": None
    }


# ─────────────────────────────────────────────
# NODE 5 — Resolution or Escalation
# ─────────────────────────────────────────────
@traceable(name="Node5_ResolutionOrEscalation")
def resolve_or_escalate(state: AgentState) -> AgentState:
    """
    Routes based on feedback.
    Yes = true deflection - log and close
    No = attempt deeper retrieval then escalate with full context
    """
    feedback = state.get("feedback", "no")
    retry_count = state.get("retry_count", 0)

    if feedback == "yes":
        print("\n✅ Case deflected successfully. No ticket needed.")

        log_node_event(
            node_name="Node5_ResolutionOrEscalation",
            question=state["question"],
            details={
                "outcome": "deflected",
                "retry_count": retry_count,
                "confidence": state.get("answer_confidence", 0)
            }
        )

        return {
            **state,
            "outcome": "deflected",
            "escalation_context": None,
            "error": None
        }

    else:
        if retry_count < 1:
            print("\n🔄 Let me try a deeper search...")

            deeper_results = combined_search(
                state["question"],
                n_results=5
            )

            new_docs = deeper_results["documents"] if deeper_results["found"] else state.get("retrieved_docs", [])

            log_node_event(
                node_name="Node5_ResolutionOrEscalation",
                question=state["question"],
                details={
                    "outcome": "retry_triggered",
                    "retry_count": retry_count + 1,
                    "new_docs_retrieved": len(new_docs)
                }
            )

            return {
                **state,
                "retrieved_docs": new_docs,
                "retrieval_sources": ["langchain_docs", "support_kb"],
                "retry_count": retry_count + 1,
                "feedback": None,
                "error": None
            }

        else:
            print("\n🎫 Escalating to support engineer with full context...")

            try:
                summary_prompt = ESCALATION_SUMMARY_PROMPT.format(
                    question=state["question"],
                    tier=state.get("tier", "unknown"),
                    answer=state.get("answer", "none"),
                    sources=state.get("retrieval_sources", []),
                    retry_count=retry_count
                )
                summary_response = llm.invoke(summary_prompt)
                escalation_summary = summary_response.content.strip()
            except Exception:
                escalation_summary = (
                    f"User asked: {state['question']}. "
                    f"Agent attempted {retry_count} answers. "
                    f"User not satisfied."
                )

            escalation_context = {
                "original_question": state["question"],
                "tier": state.get("tier"),
                "answers_attempted": retry_count + 1,
                "last_answer": state.get("answer"),
                "sources_checked": state.get("retrieval_sources", []),
                "docs_reviewed": state.get("retrieved_docs", []),
                "confidence_at_escalation": state.get("answer_confidence", 0),
                "engineer_summary": escalation_summary
            }

            # In production this would:
            # 1. POST escalation_context to ticketing system API (Zendesk/Linear)
            # 2. Fire Slack alert to #support-escalations channel
            # 3. Update CRM with interaction record
            # Current implementation logs to LangSmith and returns context

            print("\n📋 Escalation Context:")
            print(f"Question: {escalation_context['original_question']}")
            print(f"Tier: {escalation_context['tier']}")
            print(f"Attempts: {escalation_context['answers_attempted']}")
            print(f"Summary: {escalation_summary}")

            log_node_event(
                node_name="Node5_ResolutionOrEscalation",
                question=state["question"],
                details={
                    "outcome": "escalated",
                    "retry_count": retry_count,
                    "confidence_at_escalation": state.get("answer_confidence", 0),
                    "sources_checked": state.get("retrieval_sources", [])
                }
            )

            return {
                **state,
                "outcome": "escalated",
                "escalation_context": escalation_context,
                "error": None
            }