from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

EventType = Literal[
    "RunStarted",
    "RunFinished",
    "AgentStarted",
    "AgentFinished",
    "ToolCallRequested",
    "ToolCallCompleted",
    "ToolCallFailed",
    "PlanCreated",
    "EvidenceItemCreated",
    "ResearchCompleted",
]


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = Field(default_factory=now_iso)
    type: EventType
    run_id: str
    trace_id: Optional[str] = None   
    span_id: Optional[str] = None  

    agent: Optional[str] = None      
    tool: Optional[str] = None       
    data: Dict[str, Any] = Field(default_factory=dict)