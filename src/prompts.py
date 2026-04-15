CLASSIFICATION_PROMPT = """You are a support question classifier for LangChain.

Classify the following question into one of three tiers:

TIER 1 - Simple conceptual or how-to question
- Questions about what something is
- Basic setup and installation questions
- General usage questions
- Examples: "What is a checkpointer?", "How do I install LangChain?"

TIER 2 - Debugging or integration question
- Questions about something not working
- Version specific issues
- Error messages and stack traces
- Integration problems
- Examples: "Why is my streaming broken after upgrade?", "My agent is throwing an error"

TIER 3 - Deep production or architecture question
- Complex multi-system architecture questions
- Performance at scale questions
- Questions requiring codebase level investigation
- Examples: "How do I architect a multi-tenant LangGraph deployment at 10k RPS?"

Question: {question}

Respond in this exact JSON format:
{{
    "tier": <1, 2, or 3>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<one sentence explaining why>"
}}

Return only the JSON. No other text."""


ANSWER_GENERATION_PROMPT = """You are a helpful LangChain technical support agent.

Answer the following question using ONLY the context provided below.
If the context does not contain enough information to answer confidently, say so clearly.

Question: {question}

Context from documentation:
{context}

Instructions:
- Give a direct, specific answer
- Include code examples if relevant
- Cite which source you used
- End with a confidence score from 0.0 to 1.0 based on how well the context answered the question

Format your response as:
ANSWER: <your answer here>
CITATIONS: <which sources you used>
CONFIDENCE: <0.0 to 1.0>"""


RETRY_PROMPT = """You are a helpful LangChain technical support agent.

The user was not satisfied with the previous answer. Try again with a different approach.

Original question: {question}

Previous answer that did not help: {previous_answer}

Additional context retrieved:
{context}

Instructions:
- Take a different angle than the previous answer
- Be more specific and concrete
- Include a working code example if possible
- If you still cannot answer confidently, say so clearly

Format your response as:
ANSWER: <your answer here>
CITATIONS: <which sources you used>
CONFIDENCE: <0.0 to 1.0>"""


ESCALATION_SUMMARY_PROMPT = """Summarize this support conversation for a human engineer.

Question: {question}
Tier: {tier}
Answer attempted: {answer}
Sources checked: {sources}
User feedback: Not resolved after {retry_count} attempt(s)

Write a brief, structured handoff note that tells the engineer:
1. What the user is trying to do
2. What was already tried
3. Where to start investigating

Keep it under 100 words."""


LLM_JUDGE_PROMPT = """You are evaluating a support agent's response quality.

Question asked: {question}
Agent's answer: {answer}
User feedback: {feedback}
Sources used: {sources}

Score the agent on these three dimensions from 1 to 5:
1. Relevance - Did the answer address the actual question?
2. Accuracy - Was the answer technically correct based on the sources?
3. Completeness - Did the answer give the user enough to solve their problem?

Respond in this exact JSON format:
{{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "overall": <1-5>,
    "reasoning": "<one sentence explanation>"
}}

Return only the JSON. No other text."""