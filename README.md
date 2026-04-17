# LangChain Support Deflection Agent

Built as part of the LangChain GTM Engineer take-home assignment. This agent is the production implementation of findings identified during firsthand testing of chat.langchain.com and support.langchain.com on April 8th, 2026.

---

## The Origin Story

In round 2 of the interview process I spent a day as a first customer testing both LangChain support products. I identified two specific gaps that directly informed what I built here.

The first gap was the missing feedback loop on support.langchain.com. After every Ask AI answer the Open Case button appears immediately with no confirmation step asking whether the answer helped. There is no way to distinguish between a user who was genuinely helped and a user who gave up and left. True deflection and false deflection look identical in the data. This means the deflection rate metric is inflated relative to actual customer outcomes.

The second gap was the absence of intent-based routing on chat.langchain.com. Every question regardless of complexity gets the same Docs Agent. A developer asking what is a checkpointer and a developer debugging a production streaming failure receive identical treatment. No routing intelligence exists to send complex debugging questions to deeper retrieval or to escalate architecture questions directly to engineers without wasting LLM calls.

This agent is the production implementation of both findings. Node 4 is the feedback loop. Nodes 1 and 2 are the routing intelligence.

---

## What This Agent Does

A five-node LangGraph agent that classifies developer support questions, retrieves relevant context from a ChromaDB knowledge base, generates answers using Gemini 2.5 Flash, confirms whether the answer helped, and either logs a true deflection or escalates with full conversation context packaged for a support engineer.

The gap it closes is the one between when a developer hits a problem and when they get a real answer, without a human stepping in.

Who uses it: developers working with LangChain and LangGraph who hit technical questions during development or production debugging.

---

## Architecture

Every question enters at Node 1. Every question exits at one of three endings.

```
Question
   |
Node 1 — Intent Classification
Gemini classifies into Tier 1, 2, or 3
Produces: tier, confidence score, reasoning
   |
Node 2 — Tiered Retrieval
Tier 1 searches langchain_docs only (fast path)
Tier 2 searches langchain_docs + support_kb (deeper path)
Tier 3 escalates immediately with no retrieval and no LLM calls wasted
   |
Node 3 — Answer Generation
Gemini generates answer from retrieved context only
Produces: answer, citations, confidence score
   |
Node 4 — Feedback Collection
"Did this solve your problem?" Yes or No
This confirmation step does not exist on support.langchain.com today
   |
Node 5 — Resolution or Escalation
Yes → true deflection logged to LangSmith, case closed
No (first time) → deeper search triggered, loops back to Node 3
No (second time) → context packager runs, escalates to engineer
```

Three ways the process ends. A Tier 3 architecture question exits at Node 2 with immediate escalation and no wasted compute. A resolved question exits at Node 5 when the user confirms yes, logged as a true deflection. An unresolved question exits at Node 5 after two attempts, escalated with the full conversation packaged for the engineer.

The primary tool is the ChromaDB semantic search retriever. It is a search wrapper that converts text queries into embeddings and returns the most semantically similar chunks from the knowledge base. It is called in Node 2 for Tier 1 searches (docs only) and Tier 2 searches (both collections simultaneously). The search is semantic rather than keyword-based, meaning it finds relevant content even when the user's phrasing differs from how the content is written.

---

## State Definition

The agent uses a TypedDict called AgentState that travels through every node. Each field is named, typed, and filled by a specific node. Nothing is passed as an unstructured dictionary.

```python
class AgentState(TypedDict):
    question: str                                # Input, never changes
    tier: Optional[int]                          # Filled by Node 1 (1, 2, or 3)
    classification_confidence: Optional[float]   # Filled by Node 1 (0.0 to 1.0)
    classification_reasoning: Optional[str]      # Filled by Node 1 (one sentence)
    retrieved_docs: Optional[List[str]]          # Filled by Node 2 (ChromaDB chunks)
    retrieval_sources: Optional[List[str]]       # Filled by Node 2 (collection names)
    answer: Optional[str]                        # Filled by Node 3 (generated answer)
    answer_confidence: Optional[float]           # Filled by Node 3 (0.0 to 1.0)
    citations: Optional[List[str]]               # Filled by Node 3 (source names)
    feedback: Optional[str]                      # Filled by Node 4 ("yes" or "no")
    retry_count: Optional[int]                   # Tracked across nodes
    outcome: Optional[str]                       # Filled by Node 5 ("deflected" or "escalated")
    escalation_context: Optional[dict]           # Filled by Node 5 (full handoff package)
    error: Optional[str]                         # Filled by any node that fails
```

Question is the only required field. Everything else starts as None and gets filled progressively as the question moves through the workflow. This explicit contract between nodes means a field name mismatch or type error fails loudly rather than silently.

---

## Error Handling

The agent never crashes. Every node has a fallback path.

Node 1 — if Gemini fails to return valid JSON or the API call fails, the agent defaults to Tier 2 and logs the error. The workflow continues.

Node 2 — if ChromaDB returns no results for a query, the agent returns a fallback message and continues to Node 3 which attempts a direct answer. If ChromaDB throws any exception, the agent returns a graceful fallback and continues.

Node 3 — if the Gemini API call fails, the agent retries once automatically. If the second attempt also fails, it returns a graceful error message with confidence 0.0. The workflow continues to Node 4.

Node 5 — if the escalation summary generation fails, the agent falls back to a plain text summary built directly from the state data. The escalation always completes regardless.

---

## Tech Stack

LangGraph handles agent orchestration, state management, node routing, and conditional edges. Gemini 2.5 Flash handles classification, answer generation, escalation summaries, and LLM-as-judge evaluation. ChromaDB is the persistent local vector store for docs and knowledge base content. sentence-transformers all-MiniLM-L6-v2 handles local embeddings with no API cost. LangSmith handles tracing every node execution, storing eval datasets, and monitoring. Streamlit provides an interactive UI with real-time deflection metrics. Python's logging module writes structured JSON logs for every node execution to agent.log.

---

## Knowledge Base

ChromaDB contains two collections.

langchain_docs - 43 chunks from 9 real LangChain documentation URLs plus supplementary articles authored to fill JavaScript rendering gaps. Covers persistence, streaming, state management, routing, LangSmith, tools, RAG, and a LangGraph overview.

support_kb - 7 articles authored from real resolved developer issues. Covers streaming broken after upgrade, state not persisting, ChromaDB empty results, LangSmith traces not appearing, Pydantic errors, conditional edge routing, and Gemini rate limits.

The hybrid approach was necessary because LangChain docs are JavaScript rendered — static scraping returns roughly 1 chunk per page. In production the fix is Playwright or pulling directly from LangChain's GitHub markdown source files.

---

## How to Run Locally

Prerequisites are Python 3.10 or higher, a Google AI API key with access to Gemini 2.5 Flash, and a LangSmith API key from the free tier at smith.langchain.com.

**Step 1 — Clone and set up environment**

```bash
git clone https://github.com/pranav1414/Langchain_Deflection_Agent
cd langchain-deflection-agent
py -3.10 -m venv gtmlc
gtmlc\Scripts\activate
pip install -r requirements.txt
```

**Step 2 — Configure environment variables**

Create a .env file in the project root with the following:

```
GOOGLE_API_KEY=your_gemini_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=LangChain Deflection Agent
```

**Step 3 — Populate the knowledge base**

```bash
python data/ingest.py
```

Expected output: Docs ingested 43 total chunks and Support KB ingested 7 articles.

**Step 4 — Run the Streamlit UI**

```bash
streamlit run app.py
```

Opens at http://localhost:8501

**Step 5 — Or run via CLI**

```bash
python main.py
python main.py "Why is my token streaming broken after upgrading LangGraph?"
```

**Step 6 — Run evaluation**

```bash
python tests/eval.py
```

---

## Example Inputs and Outputs

**Example 1 — Tier 1 Deflection**

Input: "How do I add a checkpointer to my LangGraph agent?"

The agent classified this as Tier 1 with 95% confidence. Node 2 searched langchain_docs only. Node 3 generated a step-by-step answer covering MemorySaver for development, SqliteSaver for local persistence, and PostgresSaver for production, with working code examples. The user confirmed it helped. Outcome: deflected. No ticket created.

**Example 2 — Tier 2 Escalation after retry**

Input: "Why is my token streaming not working after upgrading LangGraph?"

The agent classified this as Tier 2 with 95% confidence. Node 2 searched both langchain_docs and support_kb. Node 3 generated an answer identifying the astream_events version="v2" requirement with before and after code examples. The user said no. Node 5 triggered a deeper combined search with 5 results instead of 3 and generated a new more detailed answer. The user said no again. Node 5 ran the context packager and escalated with a full engineer handoff summary. Outcome: escalated after 2 attempts.

**Example 3 — Tier 3 Direct Escalation**

Input: "How do I architect a multi-tenant LangGraph deployment at 10k RPS?"

The agent classified this as Tier 3 with 100% confidence. Node 2 detected Tier 3 and exited immediately. No retrieval happened. No LLM answer was generated. The question went straight to escalation. Outcome: escalated instantly with zero wasted LLM calls.

---

## LangSmith Observability

All three example traces are live and accessible. Every node decision, every retrieval call, every LLM response, and every token cost is fully visible.

Trace 1 — Tier 1 Deflection (checkpointer question):
https://smith.langchain.com/o/e2367ab1-9b2c-4d64-bc8d-6947407d069f/projects/p/139b776a-7308-41ff-8d41-d6b2a8fe54b8?timeModel=%7B%22duration%22%3A%221d%22%7D&searchModel=%7B%7D&peek=019d8dc9-fa57-7f21-a148-2dfc8bb16b7e&peeked_trace=019d8dc9-fa57-7f21-a148-2dfc8bb16b7e

Trace 2 — Tier 2 Escalation (streaming question):
https://smith.langchain.com/o/e2367ab1-9b2c-4d64-bc8d-6947407d069f/projects/p/139b776a-7308-41ff-8d41-d6b2a8fe54b8?timeModel=%7B%22duration%22%3A%221d%22%7D&searchModel=%7B%7D&peek=019d8dd0-f60c-7ff0-a126-e6ef37c81e31&peeked_trace=019d8dd0-f60c-7ff0-a126-e6ef37c81e31

Trace 3 — Tier 3 Direct Escalation (architecture question):
https://smith.langchain.com/o/e2367ab1-9b2c-4d64-bc8d-6947407d069f/projects/p/139b776a-7308-41ff-8d41-d6b2a8fe54b8?timeModel=%7B%22duration%22%3A%221d%22%7D&searchModel=%7B%7D&peek=019d8dd5-a803-7f42-9f17-a154681e5420&peeked_trace=019d8dd5-a803-7f42-9f17-a154681e5420

---

## Evaluation Results

The evaluation script runs 15 examples across all three tiers and scores each with LLM-as-judge using Gemini 2.5 Flash. Run it with python tests/eval.py.

Tier classification accuracy was 15 out of 15 at 100%. Outcome accuracy was 15 out of 15 at 100%. Average judge score was 3.5 out of 5.0. Average answer confidence was 0.64.

Breaking down by category: streaming scored 5 out of 5, persistence scored 4 out of 5, LangSmith scored 5 out of 5, Pydantic errors scored 5 out of 5, routing scored 2 out of 5, concepts scored 1 to 2 out of 5, and installation scored 2 out of 5.

**What I learned from the eval results**

Tier classification was 100% accurate across all 15 examples. LLM-based classification outperformed what any keyword matcher could achieve because it understands intent not just words.

The low judge scores revealed a clear pattern. The knowledge base had strong coverage for debugging topics like streaming, persistence, and LangSmith but weak coverage for conceptual and definitional questions. The StateGraph example was the clearest signal: accuracy scored 5 out of 5 because the agent honestly reported it lacked sufficient context rather than hallucinating an answer. But completeness scored 1 out of 5 because the user got nothing useful.

This is the eval-to-KB feedback loop working exactly as designed. The eval results told me precisely what to add to the knowledge base. I added a LangGraph overview article, re-ran ingestion, and the "What is LangGraph" question now returns a clean definition with a code example at 100% confidence.

**Regression monitoring**

Run the eval dataset weekly. Alert if tier accuracy drops below 95%, if average judge score drops below 3.0, if average confidence drops below 0.5, or if escalation rate for Tier 1 questions exceeds 15%. Every change to prompts, KB content, or routing logic triggers an eval run before deployment. The eval dataset in LangSmith serves as the regression test suite.

---

## Key Tradeoffs

**Accuracy vs cost:** Gemini 2.5 Flash was chosen over Pro for speed and cost. Flash averages approximately $0.006 per query. At 1000 queries per day that is $6 versus roughly $60 for Pro. The tradeoff is that Flash is slightly less accurate on complex nuanced questions. The production decision would be to A/B test Flash versus Pro specifically on Tier 2 debugging questions and measure judge score improvement against cost increase.

**Latency vs depth:** Tier 1 searches docs only and typically responds in under 5 seconds. Tier 2 searches both collections and takes 2 to 3 seconds longer. Tier 3 skips all retrieval and responds instantly. The tradeoff is that Tier 2 adds latency in exchange for better answer quality on debugging questions. The production decision would be to cache frequent Tier 1 queries to reduce latency across the most common question types.

**Determinism vs flexibility:** Classification uses LLM judgment rather than keyword rules. This is flexible and handles natural language variations well. The tradeoff is that LLM classification has inherent variance. The same question could theoretically be classified differently across two runs. The production decision would be to add a confidence threshold so that if classification confidence is below 0.7, the agent defaults to Tier 2 to err on the side of deeper retrieval.

**User control vs automation:** The agent shows a low confidence warning below 60% but does not automatically escalate. The user retains control over when to open a ticket. Automatic escalation was considered but rejected because it would create surprise escalations for users who found the answer sufficient despite the low confidence score. The production experiment would be A/B testing automatic versus user-controlled escalation and measuring true deflection rate against user satisfaction scores.

**RAG strict mode vs fallback:** The agent answers only from retrieved context. Gemini cannot fall back to its own training data. This prioritises accuracy and traceability over coverage. When the KB lacks relevant content the agent gives an honest low-confidence response rather than a confident but potentially outdated answer from training data. The production decision is continuous KB expansion driven by eval failure signals and escalation trace analysis.

---

## How I Would Measure Impact in Production

True deflection rate measures the percentage of questions resolved with the user confirming yes, with a target above 70%. False deflection rate measures the percentage of deflections where the user re-contacted within 24 hours, with a target below 10%. Escalation rate by tier tracks the percentage of Tier 1 and Tier 2 questions reaching escalation, with targets of Tier 1 below 10% and Tier 2 below 30%. Median response time from question to answer displayed should be below 15 seconds. Average LLM-as-judge score across the eval dataset should stay above 4.0 out of 5.0. KB coverage gap rate measures the percentage of questions returning confidence below 0.6 and should stay below 5%. Cost per deflection is total LLM cost divided by number of true deflections and should stay below $0.02.

---

## Known Limitations

JavaScript rendering means LangChain docs return approximately 1 chunk per page from static scraping because the page content loads via JavaScript after the initial HTML response. The production fix is Playwright for full rendering or pulling directly from LangChain's GitHub markdown source files.

The KB is static and populated once at ingestion. There are no automatic updates from resolved tickets. The production fix is an automated pipeline that extracts resolved escalation solutions from LangSmith traces and ingests them into the support_kb collection.

LLM judge variance means single-pass evaluation has inherent variance and the same response can score differently across runs. The production fix is running 3 passes and averaging scores.

There is no real escalation integration. The escalation packages full context but does not POST to Zendesk or Linear or fire a Slack alert. The production fix is REST API calls to the ticketing system and a Slack webhook at the escalation point in Node 5.

Session metrics reset on Streamlit restart. Deflection rate is calculated from the current session only with no persistence. The production fix is a PostgreSQL or Redis store for metrics persistence across sessions.

Tier 3 questions receive immediate escalation with no attempt to surface related documentation. The production improvement would be surfacing related docs as context while still escalating, so the user has something useful while they wait.

## What I would change next

Adding KB coverage for low-scoring categories is the immediate priority, specifically installation guides, a StateGraph definition article, and more conditional edge examples. Running the eval three times and averaging scores would reduce LLM judge variance, which is a known limitation of single-pass evaluation. Replacing static scraping with Playwright would give 10 to 15 real chunks per page instead of the current 1. Adding category-specific retrieval prompts would improve precision. Automating KB updates from resolved escalation traces in LangSmith would close the feedback loop completely without manual intervention.

---

## Friction Log

These are genuine observations from building on LangGraph and LangSmith as a first customer.

LangGraph state typing - passing a plain dict where TypedDict was expected caused silent failures in unrelated downstream nodes. Fix: always use TypedDict with Optional for every non-required field.

LangSmith tracing - load_dotenv must be called before any LangChain import or traces silently fail with no error message.

ChromaDB n_results - querying with n_results greater than collection size throws an error instead of returning what exists. Fix: always use min(n_results, collection.count()).

Gemini model availability - gemini-1.5-flash and gemini-2.0-flash returned NOT_FOUND despite being documented as available. Fix: list available models programmatically before making API calls.

LangGraph conditional edges  - return values are case sensitive and must exactly match the mapping dictionary keys. Mismatches throw a KeyError with no helpful message.

sentence-transformers BertModel warning - UNEXPECTED key warning appears on every run. It is harmless but alarming on first encounter.

KB coverage gaps - missing conceptual overview articles surface as honest no-context responses rather than errors. Eval results are the right way to find these gaps systematically.

---

## Project Structure

```
LangChain Deflection Agent/
├── src/
│   ├── state.py          # AgentState TypedDict definition
│   ├── nodes.py          # All 5 node functions with structured JSON logging
│   ├── graph.py          # LangGraph wiring, routing logic, conditional edges
│   ├── tools.py          # ChromaDB search functions (the primary tool)
│   └── prompts.py        # All LLM prompts for each node
├── data/
│   └── ingest.py         # KB ingestion, real URL scraping plus supplementary content
├── tests/
│   └── eval.py           # 15-example eval dataset with LLM-as-judge scoring
├── assets/
│   └── langchain_logo.jpeg
├── chroma_db/            # Persisted ChromaDB vector store, auto-created on first ingest
├── app.py                # Streamlit UI with real-time deflection metrics
├── main.py               # CLI entry point
├── agent.log             # Structured JSON logs, auto-created on first run
├── requirements.txt
└── README.md
```

---

## Connecting the Dots

In round 2 I identified two gaps in LangChain's support products through a day of firsthand testing as a first customer. Finding 1 was the missing feedback loop on support.langchain.com. Finding 3 was the absent intent routing on chat.langchain.com.

Node 4 is the direct implementation of Finding 1. The confirmation step that does not exist on support.langchain.com today is the entire point of that node.

Nodes 1 and 2 together are the direct implementation of Finding 3. The classification logic and tiered retrieval are the routing intelligence that does not exist on chat.langchain.com today.

The eval dataset directly measures whether both findings were correctly implemented. 100% tier accuracy confirms the routing logic works. The judge scores identify where the KB needs to grow, which is the same feedback loop I proposed in Finding 1 of the product brief.
