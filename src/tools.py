import os
import chromadb
from chromadb.utils import embedding_functions
from langchain.tools import tool
from typing import List, Dict

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Use sentence transformers for embeddings - free, no API key needed
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get or create two collections
# Collection 1 - LangChain official docs
docs_collection = client.get_or_create_collection(
    name="langchain_docs",
    embedding_function=embedding_fn
)

# Collection 2 - Support knowledge base articles
support_collection = client.get_or_create_collection(
    name="support_kb",
    embedding_function=embedding_fn
)


def search_docs(query: str, n_results: int = 3) -> Dict:
    """
    Search official LangChain documentation.
    Used for Tier 1 and Tier 2 questions.
    """
    try:
        results = docs_collection.query(
            query_texts=[query],
            n_results=min(n_results, docs_collection.count() or 1)
        )
        
        if not results["documents"][0]:
            return {
                "documents": [],
                "sources": [],
                "found": False
            }
        
        return {
            "documents": results["documents"][0],
            "sources": results["metadatas"][0] if results["metadatas"] else [],
            "found": True
        }
    except Exception as e:
        return {
            "documents": [],
            "sources": [],
            "found": False,
            "error": str(e)
        }


def search_support_kb(query: str, n_results: int = 3) -> Dict:
    """
    Search support knowledge base articles.
    Used for Tier 2 questions - real world resolved issues.
    """
    try:
        results = support_collection.query(
            query_texts=[query],
            n_results=min(n_results, support_collection.count() or 1)
        )
        
        if not results["documents"][0]:
            return {
                "documents": [],
                "sources": [],
                "found": False
            }
        
        return {
            "documents": results["documents"][0],
            "sources": results["metadatas"][0] if results["metadatas"] else [],
            "found": True
        }
    except Exception as e:
        return {
            "documents": [],
            "sources": [],
            "found": False,
            "error": str(e)
        }


def combined_search(query: str, n_results: int = 3) -> Dict:
    """
    Search both docs and support KB simultaneously.
    Used for Tier 2 questions for maximum coverage.
    """
    docs_results = search_docs(query, n_results)
    support_results = search_support_kb(query, n_results)
    
    all_documents = docs_results["documents"] + support_results["documents"]
    all_sources = docs_results["sources"] + support_results["sources"]
    
    return {
        "documents": all_documents,
        "sources": all_sources,
        "found": bool(all_documents),
        "docs_found": docs_results["found"],
        "support_found": support_results["found"]
    }


def get_collection_stats() -> Dict:
    """
    Returns how many documents are in each collection.
    Useful for debugging and verifying ingestion worked.
    """
    return {
        "docs_count": docs_collection.count(),
        "support_kb_count": support_collection.count()
    }