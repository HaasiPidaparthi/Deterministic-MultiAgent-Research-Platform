"""
RAG (Retrieval-Augmented Generation) Module

This module provides vector database functionality for storing and retrieving
documents to augment research with internal knowledge bases.
"""

import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

from engine.schemas.evidence import EvidenceItem
from engine.tools.extract import hash_text, reliability_score, relevance_score_embed


@dataclass
class RAGConfig:
    """Configuration for RAG functionality."""
    collection_name: str = "research_knowledge_base"
    embedding_model: str = "nomic-embed-text"
    persist_directory: str = "./data/chroma_db"
    similarity_threshold: float = 0.7
    max_results: int = 5
    min_relevance: float = 0.3


@dataclass
class RAGRetriever:
    """Handles vector database operations for RAG."""

    config: RAGConfig = field(default_factory=RAGConfig)
    vectorstore: Optional[Chroma] = None
    embeddings: Optional[Embeddings] = None

    def __post_init__(self):
        """Initialize the vector database and embeddings."""
        self._init_embeddings()
        self._init_vectorstore()

    def _init_embeddings(self):
        """Initialize the embedding model."""
        if "nomic" in self.config.embedding_model.lower():
            try:
                self.embeddings = OllamaEmbeddings(
                    model=self.config.embedding_model,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                )
            except Exception as exc:
                logger.warning(
                    "OllamaEmbeddings init failed for model %s: %s. Falling back to SentenceTransformerEmbeddings.",
                    self.config.embedding_model,
                    exc,
                )
                self.embeddings = SentenceTransformerEmbeddings(
                    model_name=FALLBACK_EMBEDDING_MODEL
                )
        else:
            try:
                self.embeddings = SentenceTransformerEmbeddings(
                    model_name=self.config.embedding_model
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to initialize embedding model '{self.config.embedding_model}'. "
                    "Verify that the model name is valid and required packages are installed."
                ) from exc

    def _init_vectorstore(self):
        """Initialize the Chroma vector store."""
        # Ensure persist directory exists
        os.makedirs(self.config.persist_directory, exist_ok=True)

        # Initialize Chroma client
        chroma_client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.vectorstore = Chroma(
            client=chroma_client,
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
        )

    def add_documents(self, documents: List[Document], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add documents to the vector database.

        Args:
            documents: List of LangChain Document objects
            metadata: Optional metadata to attach to all documents

        Returns:
            List of document IDs
        """
        if metadata:
            documents = [
                Document(
                    page_content=doc.page_content,
                    metadata={**(dict(doc.metadata) if doc.metadata else {}), **metadata},
                )
                for doc in documents
            ]

        documents_to_add = []
        for document in documents:
            documents_to_add.extend(_split_document(document, chunk_size=1000))

        return self.vectorstore.add_documents(documents_to_add)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add raw texts to the vector database.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts

        Returns:
            List of document IDs
        """
        return self.vectorstore.add_texts(texts, metadatas=metadatas)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search against the vector database.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Maximum distance threshold for Chroma.

        Returns:
            List of (Document, distance) tuples, where lower distance indicates a better match.
        """
        k = k or self.config.max_results
        score_threshold = score_threshold if score_threshold is not None else self.config.similarity_threshold

        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            **kwargs
        )

        if isinstance(self.vectorstore, Chroma):
            filtered_results = [
                (doc, distance) for doc, distance in docs_and_scores
                if distance <= score_threshold
            ]
        else:
            filtered_results = [
                (doc, score) for doc, score in docs_and_scores
                if score >= score_threshold
            ]

        return filtered_results

    def search_and_convert_to_evidence(
        self,
        query: str,
        question_context: str = "",
        min_relevance: Optional[float] = None
    ) -> List[EvidenceItem]:
        """
        Search the vector database and convert results to EvidenceItem format.

        Args:
            query: Search query
            question_context: Original research question for relevance scoring
            min_relevance: Minimum relevance threshold

        Returns:
            List of EvidenceItem objects
        """
        min_relevance = min_relevance or self.config.min_relevance

        # Perform similarity search
        search_results = self.similarity_search(query)

        evidence_items = []
        seen_hashes = set()

        for i, (doc, score) in enumerate(search_results):
            text = doc.page_content

            # Skip if we've seen this content before
            content_hash = hash_text(text)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # Calculate relevance score using existing logic
            relevance = relevance_score_embed(
                question=question_context,
                text=text,
                title=doc.metadata.get("title", ""),
                model=self.config.embedding_model
            )

            # Skip if below relevance threshold
            if relevance < min_relevance:
                continue

            # Extract URL from metadata or use a placeholder
            url = doc.metadata.get("url", f"internal://{doc.metadata.get('source', 'rag')}/{content_hash}")

            # Calculate reliability (use domain logic if URL available)
            reliability = reliability_score(url) if "://" in url else 0.8  # Default high reliability for internal docs

            # Create snippet
            snippet = text[:400].strip().replace("\n", " ")

            evidence_item = EvidenceItem(
                id=f"RAG{i+1}",
                url=url,
                title=doc.metadata.get("title", "Internal Document"),
                publisher=doc.metadata.get("publisher", "Knowledge Base"),
                snippet=snippet,
                reliability_score=reliability,
                relevance_score=relevance,
                content_hash=content_hash,
            )

            evidence_items.append(evidence_item)

        return evidence_items

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection."""
        try:
            if self.vectorstore is None:
                raise RuntimeError("RAG vectorstore is not initialized")
            collection = getattr(self.vectorstore, "_collection", None)
            if collection is None:
                raise RuntimeError("Unable to access the underlying Chroma collection")
            count = collection.count()
            return {
                "document_count": count,
                "collection_name": self.config.collection_name,
            }
        except Exception as e:
            logger.exception("Failed to get collection stats")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            if hasattr(self.vectorstore, "delete_collection"):
                self.vectorstore.delete_collection()
            elif hasattr(self.vectorstore, "delete"):
                self.vectorstore.delete()
            else:
                raise AttributeError("Vectorstore does not support delete_collection")
            self._init_vectorstore()  # Reinitialize
            return True
        except Exception as exc:
            logger.exception("Failed to clear collection")
            return False


def create_documents_from_evidence(evidence_items: List[EvidenceItem]) -> List[Document]:
    """
    Convert EvidenceItem objects to LangChain Document objects for ingestion.

    Args:
        evidence_items: List of EvidenceItem objects

    Returns:
        List of Document objects
    """
    documents = []

    for item in evidence_items:
        metadata = {
            "url": item.url,
            "title": item.title,
            "publisher": item.publisher,
            "reliability_score": item.reliability_score,
            "relevance_score": item.relevance_score,
            "content_hash": item.content_hash,
            "source": "evidence_ingestion",
        }

        document = Document(
            page_content=item.snippet,  # Could also use full text if available
            metadata=metadata
        )

        documents.append(document)

    return documents


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split a long text into multiple chunks without breaking words if possible."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end if end > start else start + chunk_size

    return chunks


def _split_document(document: Document, chunk_size: int = 1000) -> List[Document]:
    """Split a single Document into smaller chunks for embedding."""
    if not document.page_content or len(document.page_content) <= chunk_size:
        return [document]

    chunks = _chunk_text(document.page_content, chunk_size)
    original_metadata = dict(document.metadata) if document.metadata else {}
    split_documents: List[Document] = []

    for index, chunk in enumerate(chunks):
        metadata = {**original_metadata, "chunk_id": index}
        split_documents.append(Document(page_content=chunk, metadata=metadata))

    return split_documents


def create_documents_from_files(file_paths: List[str], chunk_size: int = 1000) -> List[Document]:
    """
    Create documents from text files with chunking.

    Args:
        file_paths: List of file paths to process
        chunk_size: Size of text chunks

    Returns:
        List of Document objects
    """
    documents = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            current_chunk = ""
            chunk_id = 0

            for paragraph in paragraphs:
                if len(paragraph) > chunk_size:
                    if current_chunk:
                        documents.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={
                                "source": file_path,
                                "chunk_id": chunk_id,
                                "title": os.path.basename(file_path),
                            },
                        ))
                        chunk_id += 1
                        current_chunk = ""

                    for chunk in _chunk_text(paragraph, chunk_size):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": file_path,
                                "chunk_id": chunk_id,
                                "title": os.path.basename(file_path),
                            },
                        ))
                        chunk_id += 1
                    continue

                if current_chunk and len(current_chunk) + len(paragraph) + 2 > chunk_size:
                    documents.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            "source": file_path,
                            "chunk_id": chunk_id,
                            "title": os.path.basename(file_path),
                        },
                    ))
                    chunk_id += 1
                    current_chunk = paragraph
                else:
                    current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph

            if current_chunk.strip():
                documents.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={
                        "source": file_path,
                        "chunk_id": chunk_id,
                        "title": os.path.basename(file_path),
                    },
                ))

        except Exception as exc:
            logger.exception("Failed to process file %s", file_path)
            continue

    return documents