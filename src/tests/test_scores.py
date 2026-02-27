import pytest

import engine.tools.extract as ex


def _ollama_available() -> bool:
    # ex.ollama is set to None if import failed in extract.py
    if getattr(ex, "ollama", None) is None:
        return False
    try:
        # Minimal sanity call: list local models
        ex.ollama.list()
        return True
    except Exception:
        return False

pytestmark = pytest.mark.integration


def test_reliability_score_high_medium_low():
    assert ex.reliability_score("https://sec.gov/foo") >= 0.85
    assert ex.reliability_score("https://oecd.org/bar") >= 0.85
    assert ex.reliability_score("https://reuters.com/a") >= 0.70
    assert ex.reliability_score("https://randomblog.example/post") == 0.52


def test_relevance_score_embed_fallback(monkeypatch):
    # Force embeddings to "not available"
    monkeypatch.setattr(ex, "ollama", None)

    q = "SMB payroll market growth"
    text = "This report discusses payroll software growth for small businesses."
    title = "Payroll software market report"

    # fallback to heuristic relevance_score and stay within [0,1]
    r = ex.relevance_score_embed(q, text, title, model="nomic-embed-text")
    assert 0.0 <= r <= 1.0
    assert r >= 0.1  # heuristic floor


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not available/running")
def test_relevance_score_embed_real_embeddings():
    q = "SMB payroll compliance requirements and market growth"
    relevant_title = "Payroll compliance and SMB payroll software growth"
    relevant_text = (
        "This report discusses payroll compliance requirements for small businesses, "
        "market growth drivers, and adoption of payroll software."
    )

    irrelevant_title = "Banana bread recipe"
    irrelevant_text = "A step by step banana bread recipe with baking tips and ingredients."

    r_good = ex.relevance_score_embed(q, relevant_text, relevant_title, model="nomic-embed-text")
    r_bad = ex.relevance_score_embed(q, irrelevant_text, irrelevant_title, model="nomic-embed-text")

    assert 0.0 <= r_good <= 1.0
    assert 0.0 <= r_bad <= 1.0
    assert r_good > r_bad


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not available/running")
def test_claim_confidence_embed_real_embeddings_prefers_supporting_citation():
    claim = "SMB payroll demand is rising due to compliance requirements and automation."

    supporting_doc = (
        "Payroll compliance and automation needs are increasing for SMBs, "
        "driving adoption of payroll software tools."
    )
    unrelated_doc = "Banana bread recipes and kitchen baking techniques for beginners."

    c_support = ex.claim_confidence_embed(
        claim_text=claim,
        cited_texts=[supporting_doc],
        cited_reliabilities=[0.8],
        model="nomic-embed-text",
    )
    c_unrelated = ex.claim_confidence_embed(
        claim_text=claim,
        cited_texts=[unrelated_doc],
        cited_reliabilities=[0.8],
        model="nomic-embed-text",
    )

    assert 0.0 <= c_support <= 1.0
    assert 0.0 <= c_unrelated <= 1.0
    assert c_support > c_unrelated


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not available/running")
def test_claim_confidence_embed_real_embeddings_rewards_more_citations():
    claim = "Payroll software adoption is increasing in small businesses."

    doc1 = "Small businesses increasingly adopt payroll software for efficiency and compliance."
    doc2 = "Payroll platforms are being adopted by SMBs to reduce administrative burden."

    c1 = ex.claim_confidence_embed(
        claim_text=claim,
        cited_texts=[doc1],
        cited_reliabilities=[0.75],
        model="nomic-embed-text",
    )
    c2 = ex.claim_confidence_embed(
        claim_text=claim,
        cited_texts=[doc1, doc2],
        cited_reliabilities=[0.75, 0.75],
        model="nomic-embed-text",
    )

    assert 0.0 <= c1 <= 1.0
    assert 0.0 <= c2 <= 1.0
    assert c2 >= c1