from typing import TypedDict, List, Dict, Any

from engine.schemas.planner import ResearchPlan
from engine.schemas.evidence import EvidenceItem
from engine.schemas.verify import VerificationReport
from engine.schemas.brief import BriefDraft


class MetricsState(TypedDict, total=False):
    start_ts: float
    elapsed_s: float

    # cost + usage
    cost_usd: float
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_total_tokens: int

    # tools
    tool_calls: Dict[str, int]         
    rejected_counts: Dict[str, int] 


class WorkflowState(TypedDict, total=False):
    question: str
    budget_usd: float
    time_limit_s: int

    iter: int  # loop counter

    plan: ResearchPlan
    evidence: List[EvidenceItem]
    brief: BriefDraft
    report: VerificationReport

    # retry knobs
    refetch_urls: List[str]
    researcher_overrides: Dict[str, Any]
    synthesizer_mode: str  # "normal" | "strict"

    metrics: MetricsState
    stop_reason: str