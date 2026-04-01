"""
RAG Database Population Script

This script provides utilities to populate the vector database with documents
for use in RAG (Retrieval-Augmented Generation) functionality.

Usage:
    python populate_rag.py --files doc1.txt doc2.pdf --urls "https://example.com/doc1" "https://example.com/doc2"
    python populate_rag.py --clear  # Clear the database
    python populate_rag.py --stats  # Show database statistics
"""

import argparse
import os
import sys
import json
import yaml
from typing import List, Optional, Dict, Any
from pathlib import Path

from engine.tools.rag import RAGRetriever, RAGConfig, create_documents_from_files
from engine.tools.web_fetch import fetch_url
from langchain_core.documents import Document

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT_DIR = Path(__file__).resolve().parents[3]
AI_BUSINESS_USE_CASES_PATH = ROOT_DIR / "data" / "ai_business_use_cases.json"


def load_business_use_case_sources(path: Path = AI_BUSINESS_USE_CASES_PATH) -> List[Dict[str, Any]]:
    """Load curated AI business use case sources from JSON."""
    if not path.exists():
        print(f"✗ Business use case source file not found: {path}")
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of source objects")

        return data
    except Exception as e:
        print(f"✗ Failed to load business use case sources from {path}: {e}")
        return []


def populate_from_files(rag: RAGRetriever, file_paths: List[str]) -> int:
    """Populate RAG database from local files."""
    print(f"Processing {len(file_paths)} files...")

    documents = create_documents_from_files(file_paths)
    if not documents:
        print("No documents created from files.")
        return 0

    print(f"Created {len(documents)} document chunks.")

    # Add to vector database
    ids = rag.add_documents(documents)
    print(f"Added {len(ids)} documents to vector database.")

    return len(ids)



def populate_business_use_case_sources(rag: RAGRetriever) -> int:
    """Populate RAG database with curated AI business use case sources."""
    print("Adding curated AI business use case sources to the RAG database...")

    sources = load_business_use_case_sources()
    if not sources:
        print("No curated AI business use case sources were loaded.")
        return 0

    documents = []
    for item in sources:
        documents.append(Document(
            page_content=item.get("summary", ""),
            metadata={
                "title": item.get("title", "Untitled Source"),
                "url": item.get("url", ""),
                "publisher": item.get("publisher", ""),
                "source": item.get("source", ""),
                "domain": item.get("domain", "ai_business_industry"),
                "use_case": item.get("use_case", "ai_business_industry"),
            }
        ))

    print(f"Fetching {len(sources)} curated business use case source pages...")

    fetched_documents = []
    fetched_count = 0
    failed_count = 0

    for item in sources:
        try:
            result = fetch_url.invoke({
                "url": item["url"],
                "extract_depth": "basic",
                "format": "text"
            })
            if result.get("status_code") == 200 and result.get("text"):
                fetched_documents.append(Document(
                    page_content=result["text"],
                    metadata={
                        "url": result.get("url", item["url"]),
                        "title": result.get("title", item["title"]),
                        "publisher": result.get("publisher", item["publisher"]),
                        "source": item["source"],
                        "domain": item["domain"],
                        "use_case": "ai_business_industry",
                    },
                ))
                fetched_count += 1
                print("✔", end="", flush=True)
            else:
                failed_count += 1
                print("✗", end="", flush=True)
        except Exception:
            failed_count += 1
            print("✗", end="", flush=True)

    print("\n")
    print(f"Fetched {fetched_count}/{len(sources)} source pages. Queued {len(fetched_documents)} page documents.")

    all_documents = documents + fetched_documents
    print(f"Preparing to add {len(all_documents)} documents to the vector database...")
    ids = rag.add_documents(all_documents)
    print(f"Added {len(ids)} vectors to the database from curated AI business use cases ({len(documents)} summary docs + {len(fetched_documents)} fetched docs).")
    return len(ids)


def populate_from_urls(rag: RAGRetriever, urls: List[str]) -> int:
    """Populate RAG database from web URLs."""
    print(f"Processing {len(urls)} URLs...")

    documents = []

    for url in urls:
        print(f"Fetching: {url}")
        try:
            result = fetch_url.invoke({
                "url": url,
                "extract_depth": "basic",
                "format": "text"
            })
            if result.get("status_code") == 200 and result.get("text"):
                # Create document
                metadata = {
                    "url": url,
                    "title": result.get("title", ""),
                    "publisher": result.get("publisher", ""),
                    "source": "web_ingestion",
                }

                document = Document(
                    page_content=result["text"],
                    metadata=metadata
                )

                documents.append(document)
                print(f"✓ Processed: {result.get('title', url)}")
            else:
                print(f"✗ Failed to fetch: {url} (status: {result.get('status_code')})")

        except Exception as e:
            print(f"✗ Error processing {url}: {e}")

    if not documents:
        print("No documents created from URLs.")
        return 0

    # Add to vector database
    ids = rag.add_documents(documents)
    print(f"Added {len(ids)} documents to vector database.")

    return len(ids)


def show_stats(rag: RAGRetriever):
    """Show vector database statistics."""
    stats = rag.get_collection_stats()
    print("=== Vector Database Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


def clear_database(rag: RAGRetriever):
    """Clear all documents from the vector database."""
    confirm = input("Are you sure you want to clear the entire vector database? (yes/no): ")
    if confirm.lower() == "yes":
        rag.clear_collection()
        print("Vector database cleared.")
    else:
        print("Operation cancelled.")


def main():
    parser = argparse.ArgumentParser(description="Populate RAG vector database")
    parser.add_argument("--files", nargs="*", help="Local files to add to the database")
    parser.add_argument("--urls", nargs="*", help="URLs to fetch and add to the database")
    parser.add_argument("--usecases", action="store_true", help="Add curated AI business use case sources")
    parser.add_argument("--clear", action="store_true", help="Clear the vector database")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--config", default="./config.yaml", help="Configuration file path")

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {args.config}: {e}")
        print("Using default RAG configuration...")
        config = {}

    # Create RAG configuration
    rag_config = RAGConfig(
        collection_name=config.get("researcher", {}).get("rag", {}).get("collection_name", "research_knowledge_base"),
        embedding_model=config.get("researcher", {}).get("rag", {}).get("embedding_model", "nomic-embed-text"),
        persist_directory=config.get("researcher", {}).get("rag", {}).get("persist_directory", "./data/chroma_db"),
    )

    # Initialize RAG retriever
    try:
        rag = RAGRetriever(rag_config)
        print("✓ RAG retriever initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize RAG retriever: {e}")
        return 1

    # Execute requested operations
    if args.clear:
        clear_database(rag)
    elif args.stats:
        show_stats(rag)
    else:
        total_added = 0

        if args.files:
            total_added += populate_from_files(rag, args.files)

        if args.urls:
            total_added += populate_from_urls(rag, args.urls)

        if args.usecases:
            total_added += populate_business_use_case_sources(rag)

        if not any([args.files, args.urls, args.usecases]):
            print("No operation specified. Use --files, --urls, --usecases, --clear, or --stats.")
            parser.print_help()
            return 1

        print(f"\n=== Summary ===")
        print(f"Total documents added: {total_added}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
