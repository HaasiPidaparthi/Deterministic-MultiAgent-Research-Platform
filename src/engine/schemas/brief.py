from typing import List
from pydantic import BaseModel, Field, ConfigDict


class Claim(BaseModel):
    text: str = Field(min_length=10)
    citations: List[str] = Field(default_factory=list, description="Evidence IDs like ['S1','S3']")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)

class BriefDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=5)
    executive_summary: str = Field(min_length=10)
    key_findings: List[Claim] = Field(min_length=3, max_length=12)
    risks: List[Claim] = Field(default_factory=list)
    recommendation: str = Field(min_length=5)
    next_steps: List[str] = Field(default_factory=list)

    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)