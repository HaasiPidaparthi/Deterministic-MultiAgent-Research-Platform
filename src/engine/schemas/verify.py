from typing import List, Optional, Literal
from pydantic import BaseModel, Field

Severity = Literal["info", "warning", "error"]

class VerificationIssue(BaseModel):
    severity: Severity
    code: str
    message: str
    location: Optional[str] = None  # e.g. "key_findings[2]" or "risks[0]"
    evidence_ids: List[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    passed: bool
    claim_count: int
    cited_claim_count: int
    citation_coverage: float = Field(ge=0.0, le=1.0)

    min_sources_required: int
    sources_used: int
    min_reliability_required: float
    min_reliability_observed: float

    issues: List[VerificationIssue] = Field(default_factory=list)