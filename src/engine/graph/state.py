from typing import TypedDict, List

from engine.schemas.planner import ResearchPlan
from engine.schemas.evidence import EvidenceItem
from engine.schemas.verify import VerificationReport
from engine.schemas.brief import BriefDraft

class WorkflowState(TypedDict, total=False):
    question: str
    budget_usd: float
    time_limit_s: int

    plan: ResearchPlan
    evidence: List[EvidenceItem]
    brief: BriefDraft
    report: VerificationReport