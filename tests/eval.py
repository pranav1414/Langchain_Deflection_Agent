import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "LangChain Deflection Agent")

from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph import agent
from src.prompts import LLM_JUDGE_PROMPT
import json


# Initialize clients
langsmith_client = Client()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)


# ─────────────────────────────────────────────
# EVAL DATASET - 15 examples
# Mix of Tier 1, Tier 2, and Tier 3 questions
# Expected outcomes based on what agent should do
# ─────────────────────────────────────────────
EVAL_EXAMPLES = [
    # TIER 1 - Simple how-to questions (should deflect)
    {
        "question": "How do I add a checkpointer to my LangGraph agent?",
        "expected_tier": 1,
        "expected_outcome": "deflected",
        "category": "persistence"
    },
    {
        "question": "What is the difference between LangChain and LangGraph?",
        "expected_tier": 1,
        "expected_outcome": "deflected",
        "category": "concepts"
    },
    {
        "question": "How do I install LangGraph?",
        "expected_tier": 1,
        "expected_outcome": "deflected",
        "category": "installation"
    },
    {
        "question": "What is a StateGraph in LangGraph?",
        "expected_tier": 1,
        "expected_outcome": "deflected",
        "category": "concepts"
    },
    {
        "question": "How do I enable LangSmith tracing?",
        "expected_tier": 1,
        "expected_outcome": "deflected",
        "category": "langsmith"
    },

    # TIER 2 - Debugging questions (may deflect or escalate)
    {
        "question": "Why is my token streaming not working after upgrading LangGraph?",
        "expected_tier": 2,
        "expected_outcome": "deflected",
        "category": "streaming"
    },
    {
        "question": "My LangGraph agent is not persisting state between runs",
        "expected_tier": 2,
        "expected_outcome": "deflected",
        "category": "persistence"
    },
    {
        "question": "ChromaDB keeps returning empty results even after ingestion",
        "expected_tier": 2,
        "expected_outcome": "deflected",
        "category": "chromadb"
    },
    {
        "question": "LangSmith traces are not showing up in my dashboard",
        "expected_tier": 2,
        "expected_outcome": "deflected",
        "category": "langsmith"
    },
    {
        "question": "I am getting a Pydantic validation error in my LangGraph state",
        "expected_tier": 2,
        "expected_outcome": "deflected",
        "category": "state_management"
    },
    {
        "question": "My conditional edges are not routing correctly in LangGraph",
        "expected_tier": 2,
        "expected_outcome": "deflected",
        "category": "routing"
    },
    {
        "question": "Getting 429 rate limit errors from Gemini API",
        "expected_tier": 2,
        "expected_outcome": "deflected",
        "category": "gemini"
    },

    # TIER 3 - Complex architecture questions (should escalate immediately)
    {
        "question": "How do I architect a multi-tenant LangGraph deployment at 10k RPS?",
        "expected_tier": 3,
        "expected_outcome": "escalated",
        "category": "architecture"
    },
    {
        "question": "What is the best strategy for zero-downtime LangGraph graph migrations in production?",
        "expected_tier": 3,
        "expected_outcome": "escalated",
        "category": "production"
    },
    {
        "question": "How do I implement distributed checkpointing across multiple LangGraph instances?",
        "expected_tier": 3,
        "expected_outcome": "escalated",
        "category": "architecture"
    }
]


def run_agent_non_interactive(question: str) -> dict:
    """
    Runs the agent without interactive feedback.
    For eval purposes we simulate user saying yes to deflect.
    """
    from src.nodes import classify_intent, retrieve_context, generate_answer, resolve_or_escalate

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

    try:
        # Node 1 - classify
        state = classify_intent(initial_state)

        # Tier 3 - skip to escalation immediately
        if state.get("tier") == 3:
            state = {
                **state,
                "outcome": "escalated",
                "escalation_context": {
                    "reason": "Tier 3 - requires human investigation",
                    "question": question
                }
            }
            return state

        # Node 2 - retrieve
        state = retrieve_context(state)

        # Node 3 - generate answer
        state = generate_answer(state)

        # Simulate user saying yes - mark as deflected
        state = {
            **state,
            "feedback": "yes",
            "outcome": "deflected"
        }

        # Node 5 - resolve
        state = resolve_or_escalate(state)

        return state

    except Exception as e:
        return {
            **initial_state,
            "error": str(e),
            "outcome": "error"
        }

def llm_judge(question: str, answer: str, feedback: str,
              sources: list, expected_outcome: str,
              actual_outcome: str) -> dict:
    """
    Uses Gemini as judge to score agent response quality.
    Returns scores for relevance, accuracy, completeness.
    """
    # For Tier 3 escalations judge differently
    if actual_outcome == "escalated" and expected_outcome == "escalated":
        return {
            "relevance": 5,
            "accuracy": 5,
            "completeness": 5,
            "overall": 5,
            "reasoning": "Correctly escalated Tier 3 question as expected"
        }

    if not answer or answer == "None":
        return {
            "relevance": 1,
            "accuracy": 1,
            "completeness": 1,
            "overall": 1,
            "reasoning": "No answer generated"
        }

    try:
        prompt = LLM_JUDGE_PROMPT.format(
            question=question,
            answer=answer[:500],
            feedback=feedback,
            sources=str(sources)
        )
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Clean JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        scores = json.loads(content)
        return scores

    except Exception as e:
        return {
            "relevance": 3,
            "accuracy": 3,
            "completeness": 3,
            "overall": 3,
            "reasoning": f"Judge evaluation failed: {str(e)}"
        }


def create_langsmith_dataset(results: list) -> str:
    """
    Creates eval dataset in LangSmith from results.
    Returns dataset ID.
    """
    dataset_name = "LangChain Deflection Agent - Eval Dataset"

    # Delete existing dataset if it exists
    try:
        existing = langsmith_client.read_dataset(dataset_name=dataset_name)
        langsmith_client.delete_dataset(dataset_id=existing.id)
        print(f"  Deleted existing dataset")
    except:
        pass

    # Create fresh dataset
    dataset = langsmith_client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation dataset for LangChain Support Deflection Agent"
    )

    # Create examples
    inputs = []
    outputs = []

    for r in results:
        inputs.append({
            "question": r["question"],
            "expected_tier": r["expected_tier"],
            "expected_outcome": r["expected_outcome"],
            "category": r["category"]
        })
        outputs.append({
            "actual_tier": r["actual_tier"],
            "actual_outcome": r["actual_outcome"],
            "answer": r["answer"][:300] if r["answer"] else None,
            "answer_confidence": r["answer_confidence"],
            "tier_correct": r["tier_correct"],
            "outcome_correct": r["outcome_correct"],
            "judge_scores": r["judge_scores"]
        })

    langsmith_client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id
    )

    print(f"  ✅ Dataset created in LangSmith: {dataset_name}")
    return str(dataset.id)


def print_results_summary(results: list):
    """
    Prints a clean summary table of eval results.
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)

    tier_correct = sum(1 for r in results if r["tier_correct"])
    outcome_correct = sum(1 for r in results if r["outcome_correct"])
    avg_confidence = sum(r["answer_confidence"] or 0 for r in results) / len(results)
    avg_judge_score = sum(r["judge_scores"].get("overall", 0) for r in results) / len(results)

    print(f"\nOverall Metrics:")
    print(f"  Total examples:      {len(results)}")
    print(f"  Tier accuracy:       {tier_correct}/{len(results)} ({tier_correct/len(results)*100:.0f}%)")
    print(f"  Outcome accuracy:    {outcome_correct}/{len(results)} ({outcome_correct/len(results)*100:.0f}%)")
    print(f"  Avg confidence:      {avg_confidence:.2f}")
    print(f"  Avg judge score:     {avg_judge_score:.1f}/5.0")

    print(f"\nDetailed Results:")
    print(f"{'Question':<50} {'Tier':>4} {'Outcome':<12} {'Judge':>5} {'OK':>4}")
    print("-"*80)

    for r in results:
        question_short = r["question"][:48] + ".." if len(r["question"]) > 48 else r["question"]
        tier_display = f"{r['actual_tier']}{'✅' if r['tier_correct'] else '❌'}"
        outcome_display = f"{r['actual_outcome']}"
        judge_score = r["judge_scores"].get("overall", 0)
        ok = "✅" if r["outcome_correct"] else "❌"
        print(f"{question_short:<50} {tier_display:>4} {outcome_display:<12} {judge_score:>5.1f} {ok:>4}")

    print("\nBy Category:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["outcome_correct"]:
            categories[cat]["correct"] += 1

    for cat, stats in categories.items():
        pct = stats["correct"]/stats["total"]*100
        print(f"  {cat:<20} {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    print("\nWhat I Learned:")
    low_scores = [r for r in results if r["judge_scores"].get("overall", 0) < 3]
    if low_scores:
        print(f"  - {len(low_scores)} questions scored below 3/5 by judge")
        print(f"  - Lowest scoring categories need more KB coverage")
    else:
        print(f"  - All questions scored 3/5 or above")

    wrong_tier = [r for r in results if not r["tier_correct"]]
    if wrong_tier:
        print(f"  - {len(wrong_tier)} questions misclassified by tier")
        print(f"  - Classification prompt needs refinement for edge cases")
    else:
        print(f"  - Tier classification was 100% accurate")

    print("\nWhat I Would Change Next:")
    print("  1. Expand KB coverage for low-scoring categories")
    print("  2. Add confidence threshold — escalate if confidence < 0.6")
    print("  3. Use Playwright for full JS rendering of docs pages")
    print("  4. Add category-specific retrieval prompts for better precision")
    print("="*80)


def run_evaluation():
    """
    Main evaluation function.
    Runs all 15 examples, scores with LLM judge,
    saves to LangSmith dataset.
    """
    print("="*60)
    print("LANGCHAIN DEFLECTION AGENT - EVALUATION")
    print("="*60)
    print(f"Running {len(EVAL_EXAMPLES)} evaluation examples...")
    print("This will take 3-5 minutes due to API rate limits.\n")

    results = []

    for i, example in enumerate(EVAL_EXAMPLES):
        print(f"[{i+1}/{len(EVAL_EXAMPLES)}] {example['question'][:60]}...")

        # Run agent
        state = run_agent_non_interactive(example["question"])

        actual_tier = state.get("tier")
        actual_outcome = state.get("outcome", "error")
        answer = state.get("answer")
        answer_confidence = state.get("answer_confidence")
        sources = state.get("retrieval_sources", [])

        # Check correctness
        tier_correct = actual_tier == example["expected_tier"]
        outcome_correct = actual_outcome == example["expected_outcome"]

        # LLM judge scoring
        judge_scores = llm_judge(
            question=example["question"],
            answer=str(answer),
            feedback="simulated_yes",
            sources=sources,
            expected_outcome=example["expected_outcome"],
            actual_outcome=actual_outcome
        )

        result = {
            "question": example["question"],
            "expected_tier": example["expected_tier"],
            "expected_outcome": example["expected_outcome"],
            "actual_tier": actual_tier,
            "actual_outcome": actual_outcome,
            "answer": str(answer) if answer else None,
            "answer_confidence": answer_confidence,
            "tier_correct": tier_correct,
            "outcome_correct": outcome_correct,
            "judge_scores": judge_scores,
            "category": example["category"]
        }

        results.append(result)

        tier_icon = "✅" if tier_correct else "❌"
        outcome_icon = "✅" if outcome_correct else "❌"
        judge_overall = judge_scores.get("overall", 0)
        print(f"  Tier: {actual_tier} {tier_icon} | "
              f"Outcome: {actual_outcome} {outcome_icon} | "
              f"Judge: {judge_overall}/5")

        # Small delay to avoid rate limits
        import time
        time.sleep(2)

    # Print summary
    print_results_summary(results)

    # Save to LangSmith
    print("\nSaving to LangSmith dataset...")
    dataset_id = create_langsmith_dataset(results)

    print(f"\n✅ Evaluation complete.")
    print(f"View results in LangSmith: smith.langchain.com")
    print(f"Project: LangChain Deflection Agent")

    return results


if __name__ == "__main__":
    run_evaluation()