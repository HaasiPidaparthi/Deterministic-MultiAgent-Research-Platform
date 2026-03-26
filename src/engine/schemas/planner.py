from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

class StopCriteria(BaseModel):
    min_sources: int = Field(default=1, ge=1, description="Minimum distinct sources to collect")
    min_claim_coverage: float = Field(default=0.85, ge=0.0, le=1.0, description="Fraction of claims supported by evidence")
    max_minutes: Optional[int] = Field(default=None, ge=1, description="Optional time cap for research stage")

class SubQuestion(BaseModel):
    question: str = Field(min_length=1)
    search_queries: List[str] = Field(default_factory=list, min_length=0, max_length=50)

class Assumption(BaseModel):
    assumption: str = Field(min_length=5)
    rationale: Optional[str] = None


class ResearchPlan(BaseModel):
    # Ignore extra keys like "research_question", "scope", "constraints"
    model_config = ConfigDict(extra="ignore")

    # Accept strings or objects; normalize to objects after
    subquestions: List[Union[str, SubQuestion]] = Field(min_length=3, max_length=12)
    
    # Keep a flat list of search queries for downstream use
    search_queries: List[str] = Field(default_factory=list, min_length=0, max_length=50)
    
    stop_criteria: StopCriteria = Field(default_factory=StopCriteria)
    
    # Accept strings or objects; normalize to objects after
    assumptions: List[Union[str, Assumption]] = Field(default_factory=list)
    risks_to_check: List[str] = Field(default_factory=list)

    def model_post_init(self, __context):
        # --- Normalize subquestions into SubQuestion objects ---
        normalized_sq: List[SubQuestion] = []
        for sq in self.subquestions:
            if isinstance(sq, str):
                normalized_sq.append(SubQuestion(question=sq, search_queries=[]))
            else:
                normalized_sq.append(sq)
        self.subquestions = normalized_sq

        # --- Normalize assumptions into Assumption objects ---
        normalized_as: List[Assumption] = []
        for a in self.assumptions:
            if isinstance(a, str):
                normalized_as.append(Assumption(assumption=a))
            else:
                normalized_as.append(a)
        self.assumptions = normalized_as

        # --- Ensure flat search_queries exists ---
        # If top-level search_queries is missing, flatten from subquestions
        if not self.search_queries:
            flat: List[str] = []
            for sq in self.subquestions:
                flat.extend(sq.search_queries)
            seen = set()
            self.search_queries = [q for q in flat if q and not (q in seen or seen.add(q))]
        else:
            # Clean + dedupe top-level queries
            cleaned = [q.strip() for q in self.search_queries if isinstance(q, str) and q.strip()]
            seen = set()
            self.search_queries = [q for q in cleaned if not (q in seen or seen.add(q))]
