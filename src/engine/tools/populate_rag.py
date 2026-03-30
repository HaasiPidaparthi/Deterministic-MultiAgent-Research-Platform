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
import yaml
from typing import List, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from engine.tools.rag import RAGRetriever, RAGConfig, create_documents_from_files
from engine.tools.web_fetch import fetch_url
from langchain_core.documents import Document


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


def populate_with_sample_data(rag: RAGRetriever) -> int:
    """Populate RAG database with sample business documents."""
    print("Adding sample business documents to RAG database...")

    sample_documents = [
        Document(
            page_content="""
            Business Intelligence (BI) encompasses the strategies and technologies used by enterprises
            for the data analysis of business information. BI technologies provide historical, current,
            and predictive views of business operations. Common functions of BI technologies include
            reporting, online analytical processing, analytics, data mining, process mining, complex
            event processing, business performance management, benchmarking, text mining, and predictive analytics.

            Key components of BI systems:
            1. Data warehousing: Centralized repository for business data
            2. ETL processes: Extract, Transform, Load operations
            3. OLAP cubes: Multidimensional data analysis
            4. Dashboards and reporting tools
            5. Data visualization and analytics

            BI helps organizations make better business decisions by providing actionable insights
            from their data. Modern BI systems increasingly incorporate artificial intelligence
            and machine learning capabilities.
            """,
            metadata={
                "title": "Business Intelligence Overview",
                "source": "sample_data",
                "domain": "business_intelligence",
                "url": "https://example.com/bi-overview"
            }
        ),
        Document(
            page_content="""
            Market Research is the systematic gathering, recording, and analysis of qualitative and
            quantitative data about issues relating to marketing products and services. The goal of
            market research is to identify and assess how changing elements of the marketing mix
            impacts customer behavior.

            Types of market research:
            1. Primary research: Data collected directly from respondents
            2. Secondary research: Analysis of existing data sources
            3. Quantitative research: Numerical data and statistical analysis
            4. Qualitative research: Non-numerical insights and understanding

            Key methodologies:
            - Surveys and questionnaires
            - Focus groups and interviews
            - Observational research
            - Experimental research
            - Competitive analysis

            Market research helps companies understand customer needs, evaluate market opportunities,
            and make informed business decisions. In today's digital age, market research increasingly
            incorporates big data analytics and social media monitoring.
            """,
            metadata={
                "title": "Market Research Fundamentals",
                "source": "sample_data",
                "domain": "market_research",
                "url": "https://example.com/market-research"
            }
        ),
        Document(
            page_content="""
            Artificial Intelligence in Business Applications

            AI is transforming business operations across all industries, from customer service to
            supply chain management. Key applications include:

            Customer Service:
            - Chatbots and virtual assistants for 24/7 support
            - Sentiment analysis for customer feedback
            - Predictive customer behavior modeling

            Sales and Marketing:
            - Lead scoring and qualification
            - Personalized marketing campaigns
            - Dynamic pricing optimization
            - Customer lifetime value prediction

            Operations:
            - Predictive maintenance for equipment
            - Supply chain optimization
            - Quality control automation
            - Fraud detection and prevention

            Human Resources:
            - Resume screening and candidate matching
            - Employee engagement analysis
            - Performance prediction and coaching

            Challenges in AI adoption:
            - Data quality and availability
            - Integration with legacy systems
            - Regulatory compliance and ethics
            - Skills gap and organizational change
            - Cost of implementation and maintenance

            Future trends include explainable AI, edge computing, and autonomous systems.
            """,
            metadata={
                "title": "AI Applications in Business",
                "source": "sample_data",
                "domain": "artificial_intelligence",
                "url": "https://example.com/ai-business"
            }
        ),
        Document(
            page_content="""
            Competitive Analysis Framework

            Competitive analysis is a critical component of strategic planning that helps organizations
            understand their position in the market relative to competitors. A comprehensive framework
            includes:

            1. Competitor Identification:
               - Direct competitors: Companies offering similar products/services
               - Indirect competitors: Companies addressing the same customer needs differently
               - Potential entrants: New companies that could enter the market

            2. Competitor Profiling:
               - Market share and growth trends
               - Product/service offerings and pricing
               - Target customer segments
               - Distribution channels and partnerships
               - Marketing strategies and brand positioning

            3. SWOT Analysis for each competitor:
               - Strengths: What they do well
               - Weaknesses: Areas of vulnerability
               - Opportunities: Market gaps they could exploit
               - Threats: External factors that could harm them

            4. Strategic Positioning:
               - Cost leadership vs. differentiation strategies
               - Market segmentation and targeting
               - Value proposition analysis

            5. Benchmarking:
               - Performance metrics comparison
               - Best practices identification
               - Gap analysis and improvement opportunities

            Effective competitive analysis requires both quantitative data (market share, financials)
            and qualitative insights (customer perceptions, employee feedback).
            """,
            metadata={
                "title": "Competitive Analysis Framework",
                "source": "sample_data",
                "domain": "strategy",
                "url": "https://example.com/competitive-analysis"
            }
        ),
        Document(
            page_content="""
            Data Privacy and Compliance in the Digital Age

            As businesses increasingly rely on data-driven decision making, compliance with data
            privacy regulations has become a critical concern. Key regulations include:

            GDPR (General Data Protection Regulation):
            - Applies to EU citizens' data regardless of company location
            - Requires explicit consent for data collection
            - Grants users rights to access, rectify, and delete their data
            - Imposes significant fines for non-compliance (up to 4% of global revenue)

            CCPA (California Consumer Privacy Act):
            - Gives California residents rights over their personal information
            - Requires businesses to disclose data collection practices
            - Allows users to opt-out of data sales
            - Includes private right of action for data breaches

            Other notable regulations:
            - PIPL (China): Personal Information Protection Law
            - LGPD (Brazil): Lei Geral de Proteção de Dados
            - PDPA (Singapore): Personal Data Protection Act

            Compliance challenges:
            - Cross-border data transfers
            - Third-party vendor management
            - Data mapping and inventory
            - Privacy by design implementation
            - Regular audits and assessments

            Organizations should adopt a privacy-first approach, implementing technical and
            organizational measures to protect personal data throughout its lifecycle.
            """,
            metadata={
                "title": "Data Privacy and Compliance",
                "source": "sample_data",
                "domain": "compliance",
                "url": "https://example.com/data-privacy"
            }
        )
    ]

    # Add documents to vector database
    ids = rag.add_documents(sample_documents)
    print(f"✓ Added {len(ids)} sample documents to RAG database")

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
    parser.add_argument("--sample", action="store_true", help="Add sample business documents")
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

        if args.sample:
            total_added += populate_with_sample_data(rag)

        if not args.files and not args.urls:
            print("No operation specified. Use --files, --urls, --clear, or --stats.")
            parser.print_help()
            return 1

        print(f"\n=== Summary ===")
        print(f"Total documents added: {total_added}")

    return 0


if __name__ == "__main__":
    sys.exit(main())