import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import time
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.tools import docs_collection, support_collection


# ─────────────────────────────────────────────
# REAL DOCS URLS FROM langgraph docs
# ─────────────────────────────────────────────
DOCS_URLS = [
    "https://langchain-ai.github.io/langgraph/concepts/persistence/",
    "https://langchain-ai.github.io/langgraph/concepts/streaming/",
    "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "https://langchain-ai.github.io/langgraph/how-tos/persistence/",
    "https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/",
    "https://langchain-ai.github.io/langgraph/how-tos/tool-calling/",
    "https://docs.smith.langchain.com/observability/how_to_guides/tracing/trace_with_langgraph",
    "https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/",
    "https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/",
]


# ─────────────────────────────────────────────
# SUPPLEMENTARY DOCS CONTENT
# Manually authored for topics where JS rendering
# returns thin content. Based on real LangChain docs.
# In production: use Playwright or GitHub markdown files.
# ─────────────────────────────────────────────
SUPPLEMENTARY_DOCS = [
    {
        "id": "sup_001",
        "text": """LangGraph Persistence and Checkpointers - Complete Guide

LangGraph agents are stateless by default. Every invocation starts fresh
with no memory of previous runs. To enable persistence you must compile
your graph with a checkpointer.

MemorySaver - for development and testing:
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

SqliteSaver - for local persistence that survives restarts:
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=checkpointer)

PostgresSaver - for production at scale:
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(POSTGRES_URI)
app = graph.compile(checkpointer=checkpointer)

Using thread_id to maintain conversation context:
config = {"configurable": {"thread_id": "user_session_123"}}
result = app.invoke({"messages": "hello"}, config)

The thread_id is critical. Using the same thread_id continues the
existing conversation. Using a different thread_id starts a new one.
Always store thread_id per user session in production.""",
        "metadata": {"source": "docs.langchain.com/supplementary", "topic": "persistence", "type": "official_doc"}
    },

    {
    "id": "sup_009",
    "text": """What is LangGraph - Overview

LangGraph is a Python framework built by LangChain for building 
stateful, multi-step AI agent applications. It extends LangChain 
by adding the ability to create cyclical, graph-based workflows 
where AI agents can make decisions, use tools, and maintain state 
across multiple steps.

Key concepts:
- Nodes: Individual steps or functions in your agent workflow
- Edges: Connections between nodes that define the flow
- State: A shared data structure that persists across all nodes
- Graph: The complete workflow connecting all nodes and edges

Why use LangGraph over plain LangChain:
- LangChain is great for linear chains
- LangGraph adds loops, branching, and stateful multi-step logic
- LangGraph handles complex agent architectures that need 
  to make decisions and retry based on results

Simple example:
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("classify", classify_node)
graph.add_node("answer", answer_node)
graph.add_edge("classify", "answer")
graph.add_edge("answer", END)
app = graph.compile()

LangGraph is used in production by companies building 
customer support agents, coding assistants, research agents,
and any multi-step AI workflow that requires state management.""",
    "metadata": {"source": "docs.langchain.com/supplementary", 
                 "topic": "overview", "type": "official_doc"}
    },

    {
        "id": "sup_002",
        "text": """LangGraph Streaming - Complete Guide

LangGraph supports multiple streaming modes for real-time token output.

Method 1 - astream for node-level streaming:
async for chunk in app.astream(input, config):
    print(chunk)

Method 2 - astream_events for token-level streaming (recommended):
async for event in app.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)

CRITICAL: Always specify version="v2" in astream_events.
The v1 format was deprecated in LangGraph 0.2+.
Not specifying version causes streaming to fail silently.

Stream modes available:
- stream_mode="values" - emits full state after each node
- stream_mode="updates" - emits only the state changes
- stream_mode="messages" - emits LLM tokens as they generate

For perceived latency improvement use streaming even when
total processing time is the same - users see progress immediately.""",
        "metadata": {"source": "docs.langchain.com/supplementary", "topic": "streaming", "type": "official_doc"}
    },
    {
        "id": "sup_003",
        "text": """LangGraph State Management - Complete Guide

State is the central data structure in LangGraph.
Every node receives the full state and returns updates to it.

Define state with TypedDict:
from typing import TypedDict, Optional, List, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only
    question: str
    answer: Optional[str]
    retry_count: int

The Annotated[list, operator.add] pattern means LangGraph
automatically appends new messages instead of replacing them.

Node pattern - return only fields you want to update:
def my_node(state: AgentState) -> AgentState:
    return {
        "answer": "the answer",
        "retry_count": state["retry_count"] + 1
    }

LangGraph merges returned dict with existing state.
You never need to return the entire state object.

For complex state use Pydantic BaseModel instead of TypedDict
for automatic validation of field types.""",
        "metadata": {"source": "docs.langchain.com/supplementary", "topic": "state_management", "type": "official_doc"}
    },
    {
        "id": "sup_004",
        "text": """LangGraph Conditional Edges and Routing - Complete Guide

Conditional edges allow your graph to branch based on state values.

Basic conditional edge pattern:
from langgraph.graph import StateGraph, END

def routing_function(state: AgentState) -> str:
    if state["feedback"] == "yes":
        return "end"
    elif state["retry_count"] < 2:
        return "retry"
    else:
        return "escalate"

graph.add_conditional_edges(
    "source_node",
    routing_function,
    {
        "end": END,
        "retry": "answer_node",
        "escalate": "escalation_node"
    }
)

The routing function must return a string that exactly matches
one of the keys in the mapping dictionary.

Common mistake: returning "END" (uppercase) instead of using
the END constant from langgraph.graph.

For simple binary routing use a lambda:
graph.add_conditional_edges(
    "node",
    lambda state: "yes" if state["done"] else "no",
    {"yes": END, "no": "retry_node"}
)""",
        "metadata": {"source": "docs.langchain.com/supplementary", "topic": "routing", "type": "official_doc"}
    },
    {
        "id": "sup_005",
        "text": """LangSmith Tracing and Evaluation - Complete Guide

LangSmith provides observability for LangChain and LangGraph applications.

Setup - set environment variables before importing LangChain:
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_key"
os.environ["LANGCHAIN_PROJECT"] = "your_project_name"

All LangGraph runs are automatically traced once these are set.

Manual tracing with @traceable decorator:
from langsmith import traceable

@traceable(name="my_custom_step")
def process_data(input: str) -> str:
    return transform(input)

Creating evaluation datasets:
from langsmith import Client
client = Client()

dataset = client.create_dataset("my_eval_dataset")
client.create_examples(
    inputs=[{"question": "What is a checkpointer?"}],
    outputs=[{"answer": "A checkpointer saves LangGraph state..."}],
    dataset_id=dataset.id
)

Running evaluations with LLM as judge:
from langsmith.evaluation import evaluate, LangChainStringEvaluator
evaluator = LangChainStringEvaluator("criteria", ...)
results = evaluate(my_agent, data=dataset, evaluators=[evaluator])""",
        "metadata": {"source": "docs.langchain.com/supplementary", "topic": "langsmith", "type": "official_doc"}
    },
    {
        "id": "sup_006",
        "text": """LangChain RAG with ChromaDB - Complete Guide

RAG (Retrieval Augmented Generation) improves LLM accuracy by
providing relevant context from a knowledge base.

Setup ChromaDB vector store:
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)

Ingest documents:
collection.upsert(
    ids=["doc_1", "doc_2"],
    documents=["document text here", "another document"],
    metadatas=[{"source": "url1"}, {"source": "url2"}]
)

Query for relevant chunks:
results = collection.query(
    query_texts=["your search query"],
    n_results=3
)
relevant_chunks = results["documents"][0]

Important: n_results cannot exceed the number of documents
in the collection. Always use min(n_results, collection.count()).""",
        "metadata": {"source": "docs.langchain.com/supplementary", "topic": "rag", "type": "official_doc"}
    },
    {
        "id": "sup_007",
        "text": """LangGraph Tools and Tool Calling - Complete Guide

Tools are functions agents call to interact with external systems.

Define a tool with the @tool decorator:
from langchain.tools import tool

@tool
def search_docs(query: str) -> str:
    "Search documentation for relevant information"
    results = vector_store.query(query_texts=[query], n_results=3)
    return "\n".join(results["documents"][0])

Bind tools to LLM:
llm_with_tools = llm.bind_tools([search_docs])

Use ToolNode for automatic tool execution in LangGraph:
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([search_docs])
graph.add_node("tools", tool_node)

Tool calling flow in LangGraph:
1. LLM decides which tool to call based on the question
2. ToolNode executes the tool with provided arguments
3. Result is added to state messages
4. LLM generates final response using tool output

Error handling in tools - always use try/except:
@tool
def safe_search(query: str) -> str:
    try:
        results = vector_store.query(query_texts=[query])
        return results["documents"][0][0]
    except Exception as e:
        return f"Search failed: {str(e)}" """,
        "metadata": {"source": "docs.langchain.com/supplementary", "topic": "tools", "type": "official_doc"}
    },
    {
        "id": "sup_008",
        "text": """LangGraph Human in the Loop - Complete Guide

Human in the loop allows agents to pause and wait for human input
before continuing execution.

Basic interrupt pattern:
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["sensitive_node"]
)

When the graph hits sensitive_node it pauses and saves state.
Resume after human review:
app.invoke(None, config)

Use case - approval workflow:
1. Agent proposes an action
2. Graph pauses at approval node
3. Human reviews and approves or rejects
4. Graph resumes with human decision in state

interrupt_after vs interrupt_before:
- interrupt_before: pause before node runs
- interrupt_after: pause after node runs, review output

This pattern is essential for production agents handling
sensitive operations like sending emails, making payments,
or modifying databases.""",
        "metadata": {"source": "docs.langchain.com/supplementary", "topic": "human_in_loop", "type": "official_doc"}
    }
]


# ─────────────────────────────────────────────
# SUPPORT KB - manually written from real issues
# (support.langchain.com requires login)
# ─────────────────────────────────────────────
SUPPORT_KB_CONTENT = [
    {
        "id": "kb_001",
        "text": """Issue: Token streaming not working after LangGraph upgrade

Symptom: Streaming was working before upgrading LangGraph.
After upgrade to 0.2+, no tokens stream - response appears all at once.

Root cause: astream_events v1 format was deprecated in LangGraph 0.2.
The version parameter is now required.

Fix: Add version="v2" to your astream_events call:

# Before (broken after upgrade)
async for event in app.astream_events(input, config):
    print(event)

# After (fixed)
async for event in app.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")

Resolved: Yes
Affects: LangGraph 0.2.0 and above""",
        "metadata": {"source": "support.langchain.com", "topic": "streaming", "type": "resolved_ticket"}
    },
    {
        "id": "kb_002",
        "text": """Issue: LangGraph agent not persisting state between runs

Symptom: Agent starts fresh every time it is invoked.
Previous conversation context is lost.

Root cause: Graph compiled without a checkpointer.
LangGraph is stateless by default.

Fix:
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(input, config)

Common mistake: Using a different thread_id each time,
which creates a new conversation thread instead of continuing.

Resolved: Yes""",
        "metadata": {"source": "support.langchain.com", "topic": "persistence", "type": "resolved_ticket"}
    },
    {
        "id": "kb_003",
        "text": """Issue: ChromaDB collection returning empty results

Symptom: query() returns empty documents list even though
collection was populated.

Root cause 1: Embedding model mismatch.
Root cause 2: n_results parameter exceeds collection size.

Fix:
count = collection.count()
results = collection.query(
    query_texts=[query],
    n_results=min(n_results, count) if count > 0 else 1
)

Resolved: Yes""",
        "metadata": {"source": "support.langchain.com", "topic": "chromadb", "type": "resolved_ticket"}
    },
    {
        "id": "kb_004",
        "text": """Issue: LangSmith traces not appearing in dashboard

Symptom: Running LangGraph agent but no traces show up in LangSmith.

Root cause: Environment variables not loaded before LangChain imports.

Fix:
from dotenv import load_dotenv
load_dotenv()  # Must be before langchain imports

Required environment variables:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=your_project_name

Resolved: Yes""",
        "metadata": {"source": "support.langchain.com", "topic": "langsmith", "type": "resolved_ticket"}
    },
    {
        "id": "kb_005",
        "text": """Issue: Pydantic validation error in LangGraph state

Symptom: ValidationError when passing state between nodes.

Fix: Use Optional for fields that may not be set initially:

from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    question: str
    answer: Optional[str]
    retrieved_docs: Optional[List[str]]

Nodes must return dict not the full state object:

def my_node(state):
    return {"answer": "hello"}

Resolved: Yes""",
        "metadata": {"source": "support.langchain.com", "topic": "state_management", "type": "resolved_ticket"}
    },
    {
        "id": "kb_006",
        "text": """Issue: LangGraph conditional edges not routing correctly

Symptom: Agent always goes to same node regardless of state.

Root cause: Routing function return value not matching
keys in conditional edges mapping dict.

Fix:
def routing_fn(state):
    if state["feedback"] == "yes":
        return "end"
    return "retry"

graph.add_conditional_edges(
    "my_node",
    routing_fn,
    {
        "end": END,
        "retry": "other_node"
    }
)

Resolved: Yes""",
        "metadata": {"source": "support.langchain.com", "topic": "routing", "type": "resolved_ticket"}
    },
    {
        "id": "kb_007",
        "text": """Issue: Google Gemini API returning 429 rate limit errors

Symptom: LangChain calls to Gemini failing with 429 Too Many Requests.

Root cause: Gemini free tier has rate limits of 15 RPM.

Fix:
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=your_key,
    max_retries=3,
    timeout=30
)

Resolved: Yes""",
        "metadata": {"source": "support.langchain.com", "topic": "gemini", "type": "resolved_ticket"}
    }
]


def scrape_page(url: str) -> str:
    """
    Scrapes text content from a single URL.
    Returns cleaned text or empty string on failure.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        for tag in soup(["nav", "footer", "script", "style", "header"]):
            tag.decompose()

        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", {"class": "md-content"})
        )
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    except Exception as e:
        print(f"  ⚠️  Failed to scrape {url}: {e}")
        return ""


def ingest_real_docs():
    """
    Scrapes real LangChain docs and combines with
    supplementary manually authored content.
    Hybrid approach for JS-rendered pages.
    """
    print("\nScraping real LangChain documentation...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    all_ids = []
    all_texts = []
    all_metadatas = []

    # Scrape real URLs
    for i, url in enumerate(DOCS_URLS):
        print(f"  Scraping ({i+1}/{len(DOCS_URLS)}): {url}")
        text = scrape_page(url)

        if not text:
            print(f"  ⚠️  Skipping empty page")
            continue

        chunks = splitter.split_text(text)
        print(f"  ✅ Got {len(chunks)} chunks")

        for j, chunk in enumerate(chunks):
            all_ids.append(f"doc_{i}_{j}")
            all_texts.append(chunk)
            all_metadatas.append({
                "source": url,
                "topic": url.split("/")[-2],
                "type": "official_doc",
                "chunk": j
            })

        time.sleep(1)

    # Add supplementary content
    print(f"\n  Adding {len(SUPPLEMENTARY_DOCS)} supplementary doc articles...")
    for doc in SUPPLEMENTARY_DOCS:
        chunks = splitter.split_text(doc["text"])
        for j, chunk in enumerate(chunks):
            all_ids.append(f"{doc['id']}_{j}")
            all_texts.append(chunk)
            all_metadatas.append(doc["metadata"])

    # Ingest into ChromaDB in batches
    if all_texts:
        batch_size = 50
        for i in range(0, len(all_texts), batch_size):
            docs_collection.upsert(
                ids=all_ids[i:i+batch_size],
                documents=all_texts[i:i+batch_size],
                metadatas=all_metadatas[i:i+batch_size]
            )
        print(f"\n✅ Docs ingested: {docs_collection.count()} total chunks")
    else:
        print("\n⚠️  No docs were scraped.")


def ingest_support_kb():
    """
    Ingests support KB articles into ChromaDB.
    """
    print("\nIngesting support knowledge base...")

    support_collection.upsert(
        ids=[kb["id"] for kb in SUPPORT_KB_CONTENT],
        documents=[kb["text"] for kb in SUPPORT_KB_CONTENT],
        metadatas=[kb["metadata"] for kb in SUPPORT_KB_CONTENT]
    )
    print(f"✅ Support KB ingested: {support_collection.count()} articles")


def ingest_all():
    """
    Main ingestion function.
    """
    print("="*60)
    print("LANGCHAIN DEFLECTION AGENT - DATA INGESTION")
    print("="*60)

    ingest_real_docs()
    ingest_support_kb()

    print("\n" + "="*60)
    print("✅ Ingestion complete. ChromaDB is ready.")
    print("="*60)


if __name__ == "__main__":
    ingest_all()