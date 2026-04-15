from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    # Input
    question: str
    
    # Classification
    tier: Optional[int]                    # 1, 2, or 3
    classification_confidence: Optional[float]
    classification_reasoning: Optional[str]
    
    # Retrieval
    retrieved_docs: Optional[List[str]]
    retrieval_sources: Optional[List[str]]
    
    # Answer
    answer: Optional[str]
    answer_confidence: Optional[float]
    citations: Optional[List[str]]
    
    # Feedback
    feedback: Optional[str]               # "yes" or "no"
    retry_count: Optional[int]
    
    # Resolution
    outcome: Optional[str]                # "deflected" or "escalated"
    escalation_context: Optional[dict]
    
    # Error handling
    error: Optional[str]