import os
import sys
from dotenv import load_dotenv

# Load environment variables BEFORE any langchain imports
load_dotenv()

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "LangChain Deflection Agent")

from src.graph import agent
from src.tools import get_collection_stats


def run_agent(question: str) -> dict:
    """
    Runs the deflection agent for a given question.
    Returns the final state after all nodes have run.
    """
    print("\n" + "="*60)
    print("LANGCHAIN SUPPORT DEFLECTION AGENT")
    print("="*60)
    print(f"Question: {question}")
    print("="*60)

    # Initial state - only question is filled
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

    # Run the agent
    final_state = agent.invoke(initial_state)

    # Print summary
    print("\n" + "="*60)
    print("AGENT SUMMARY")
    print("="*60)
    print(f"Question:  {final_state.get('question')}")
    print(f"Tier:      {final_state.get('tier')}")
    print(f"Outcome:   {final_state.get('outcome')}")
    print(f"Confidence:{final_state.get('answer_confidence')}")
    print(f"Retries:   {final_state.get('retry_count', 0)}")
    
    if final_state.get('outcome') == 'escalated':
        ctx = final_state.get('escalation_context', {})
        print(f"\nEscalation Summary:")
        print(ctx.get('engineer_summary', 'No summary available'))

    print("="*60)

    return final_state


def check_db():
    """
    Checks if ChromaDB has been populated.
    Reminds user to run ingestion if empty.
    """
    stats = get_collection_stats()
    if stats["docs_count"] == 0 and stats["support_kb_count"] == 0:
        print("\n⚠️  ChromaDB is empty.")
        print("Run this first: python data/ingest.py")
        print("Then run main.py again.\n")
        sys.exit(1)
    else:
        print(f"\n✅ ChromaDB ready — "
              f"Docs: {stats['docs_count']} | "
              f"Support KB: {stats['support_kb_count']}")


if __name__ == "__main__":
    # Check DB is populated
    check_db()

    # Example questions to test all tiers
    questions = [
        "How do I add a checkpointer to my LangGraph agent?",           # Tier 1
        "Why is my token streaming not working after upgrading LangGraph?", # Tier 2
        "How do I architect a multi-tenant LangGraph system at 10k RPS?",  # Tier 3
    ]

    # Run with first question by default
    # Change index to test different questions
    question = questions[0]

    # Or pass a custom question via command line
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    run_agent(question)