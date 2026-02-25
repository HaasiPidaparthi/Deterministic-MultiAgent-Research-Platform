from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class EvidenceItem(BaseModel):
    id: str
    url: str
    retrieved_at: str = Field(default_factory=now_iso)
    title: Optional[str] = None
    publisher: Optional[str] = None
    snippet: Optional[str] = None

    reliability_score: float = Field(default=0.5, ge=0, le=1)
    relevance_score: float = Field(default=0.5, ge=0, le=1)
    
    content_hash: Optional[str] = None