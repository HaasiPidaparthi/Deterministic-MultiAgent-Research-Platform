import uuid

import pytest
from langchain_core.documents import Document

from engine.schemas.evidence import EvidenceItem
from engine.tools.rag import (
    RAGConfig,
    RAGRetriever,
    create_documents_from_evidence,
    create_documents_from_files,
)


class DummyVectorStore:
    def __init__(self):
        self.documents = []

    def add_texts(self, texts, metadatas=None):
        ids = []
        for index, text in enumerate(texts):
            metadata = (metadatas[index] if metadatas else {}).copy()
            self.documents.append(Document(page_content=text, metadata=metadata))
            ids.append(f"doc-{len(self.documents)}")
        return ids

    def add_documents(self, documents):
        ids = []
        for document in documents:
            self.documents.append(document)
            ids.append(f"doc-{len(self.documents)}")
        return ids

    def similarity_search_with_score(self, query, k=5, **kwargs):
        query_tokens = [
            token.lower().strip('.,:;()[]{}')
            for token in query.split()
            if len(token) > 3
        ]
        results = []
        for doc in self.documents:
            score = float(
                sum(token in doc.page_content.lower() for token in query_tokens)
            )
            results.append((doc, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]

    def delete_collection(self):
        self.documents.clear()

    @property
    def _collection(self):
        return self

    def count(self):
        return len(self.documents)


@pytest.fixture
def dummy_rag(monkeypatch, tmp_path):
    monkeypatch.setattr(RAGRetriever, "_init_embeddings", lambda self: None)
    monkeypatch.setattr(RAGRetriever, "_init_vectorstore", lambda self: None)
    monkeypatch.setattr("engine.tools.rag.relevance_score_embed", lambda question, text, title, model: 0.85)

    config = RAGConfig(persist_directory=str(tmp_path / "chroma_db"))
    rag = RAGRetriever(config)
    rag.vectorstore = DummyVectorStore()
    return rag


def test_rag_add_texts_and_similarity_search(dummy_rag):
    metadata = [
        {"title": "Business Intelligence Overview", "url": "https://internal/bi", "publisher": "Internal Knowledge"},
        {"title": "Market Research Guide", "url": "https://internal/mr", "publisher": "Internal Knowledge"},
    ]
    texts = [
        "Business intelligence tools help companies analyze data and generate insights.",
        "Market research methods include surveys, interviews, and competitive analysis.",
    ]

    ids = dummy_rag.add_texts(texts, metadatas=metadata)
    assert len(ids) == 2
    assert dummy_rag.get_collection_stats()["document_count"] == 2

    results = dummy_rag.similarity_search("business intelligence tools", k=2)
    assert len(results) == 1
    doc, score = results[0]
    assert score >= 1.0
    assert doc.metadata["title"] == "Business Intelligence Overview"


def test_rag_search_and_convert_to_evidence(dummy_rag):
    metadata = [
        {
            "title": "AI in Business",
            "url": "https://internal/ai-business",
            "publisher": "Internal Knowledge",
        }
    ]
    texts = [
        "Artificial intelligence in business improves decision making and automation."
    ]

    dummy_rag.add_texts(texts, metadatas=metadata)
    evidence = dummy_rag.search_and_convert_to_evidence(
        query="artificial intelligence in business",
        question_context="How can business use artificial intelligence?",
        min_relevance=0.1,
    )

    assert len(evidence) == 1
    item = evidence[0]
    assert item.id.startswith("RAG")
    assert item.title == "AI in Business"
    assert item.url == "https://internal/ai-business"
    assert item.reliability_score >= 0.0
    assert item.relevance_score >= 0.1
    assert item.content_hash


def test_create_documents_from_evidence():
    evidence_items = [
        EvidenceItem(
            id="E1",
            url="https://example.com/article",
            title="Example Article",
            publisher="Example Publisher",
            snippet="This is a useful excerpt from the evidence article.",
            reliability_score=0.75,
            relevance_score=0.85,
            content_hash=uuid.uuid4().hex,
        )
    ]

    documents = create_documents_from_evidence(evidence_items)
    assert len(documents) == 1
    assert documents[0].page_content == evidence_items[0].snippet
    assert documents[0].metadata["title"] == evidence_items[0].title
    assert documents[0].metadata["url"] == evidence_items[0].url


def test_create_documents_from_files(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text(
        "First paragraph of the document.\n\nSecond paragraph with more content.",
        encoding="utf-8",
    )

    documents = create_documents_from_files([str(file_path)], chunk_size=30)
    assert len(documents) >= 2

    for index, document in enumerate(documents):
        assert document.metadata["source"] == str(file_path)
        assert document.metadata["title"] == "example.txt"
        assert document.metadata["chunk_id"] == index


def test_add_documents_splits_long_documents(dummy_rag):
    long_text = "word " * 500
    document = Document(
        page_content=long_text,
        metadata={"title": "Long Document", "url": "https://example.com/long"},
    )

    ids = dummy_rag.add_documents([document])
    assert len(ids) > 1
    assert dummy_rag.get_collection_stats()["document_count"] == len(ids)
    assert all("chunk_id" in doc.metadata for doc in dummy_rag.vectorstore.documents)


def test_clear_collection(dummy_rag):
    dummy_rag.add_texts(["one", "two"])
    assert dummy_rag.get_collection_stats()["document_count"] == 2
    assert dummy_rag.clear_collection() is True
    assert dummy_rag.get_collection_stats()["document_count"] == 0
