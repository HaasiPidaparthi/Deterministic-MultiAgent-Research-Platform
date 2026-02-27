import json
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from engine.schemas.brief import BriefDraft
from engine.schemas.evidence import EvidenceItem
from engine.schemas.planner import ResearchPlan
from engine.events.emitter import Emitter


SYNTH_SYSTEM = """You are a senior analyst writing an executive brief.

Return ONLY valid JSON (no markdown, no commentary) with EXACTLY these top-level keys:
- title (string)
- executive_summary (string)
- key_findings (array of objects: {{text, citations, confidence}})
- risks (array of objects: {{text, citations, confidence}})
- recommendation (string)
- next_steps (array of strings)
- assumptions (array of strings)
- limitations (array of strings)

Rules:
- Use snake_case keys exactly as written.
- Do NOT include any additional keys.
- Every item in key_findings and risks MUST include citations as evidence IDs like ["S1","S2"].
- Citations must be separate fields, not embedded in text.
- Use ONLY provided evidence. If unsupported, put it in limitations.

Output JSON only.
"""

SYNTH_HUMAN = """
Research question:
{question}

Plan:
{subquestions}

Assumptions:
{assumptions}

Risks to check:
{risks_to_check}

Evidence items (use citations like S1, S2):
{evidence_block}

Write the BriefDraft JSON now.
"""

REPAIR_SYSTEM = """Your previous output was rejected because it was not valid JSON or did not match the required schema.

Now output ONLY valid JSON with EXACTLY the required keys and correct snake_case.
No extra keys. No trailing text. No citations inside strings.
"""


def _format_evidence(evidence: List[EvidenceItem]) -> str:
    lines = []
    for e in evidence:
        lines.append(
            f"{e.id} | {e.title or ''} | {e.url} | rel={e.reliability_score:.2f} rev={e.relevance_score:.2f}\n"
            f"snippet: {e.snippet or ''}"
        )
    return "\n\n".join(lines)

def _to_brief(msg) -> BriefDraft:
    txt = msg.content.strip()
    data = json.loads(txt)
    return BriefDraft.model_validate(data)

@dataclass
class SynthesizerAgent:
    llm: BaseChatModel

    def synthesize(
        self,
        question: str,
        plan: ResearchPlan,
        evidence: List[EvidenceItem],
        emitter: Optional[Emitter] = None,
    ) -> BriefDraft:
        emitter and emitter.emit("AgentStarted", agent="synthesizer", question=question, evidence_count=len(evidence))

        prompt = ChatPromptTemplate.from_messages([("system", SYNTH_SYSTEM), ("human", SYNTH_HUMAN)])

        # JSON-only output (more reliable than tool calling for many open models)
        llm_json = self.llm.bind(response_format={"type": "json_object"}, temperature=0)

        try:
            runnable = prompt | llm_json | RunnableLambda(_to_brief)

            subq = [getattr(s, "question", str(s)) for s in plan.subquestions]
            assumptions = [getattr(a, "assumption", str(a)) for a in getattr(plan, "assumptions", [])]

            brief = runnable.invoke({
                "question": question,
                "subquestions": subq,
                "assumptions": assumptions,
                "risks_to_check": getattr(plan, "risks_to_check", []),
                "evidence_block": _format_evidence(evidence),
            })

            emitter and emitter.emit(
                "AgentFinished",
                agent="synthesizer",
                key_findings=len(brief.key_findings),
                risks=len(brief.risks),
            )
            return brief
        except Exception:
            # one retry with stricter system message
            prompt2 = ChatPromptTemplate.from_messages([
                ("system", SYNTH_SYSTEM + "\n\n" + REPAIR_SYSTEM),
                ("human", SYNTH_HUMAN),
            ])
            runnable2 = prompt2 | llm_json | RunnableLambda(_to_brief)
            brief = runnable2.invoke({
                "question": question,
                "subquestions": subq,
                "assumptions": assumptions,
                "risks_to_check": getattr(plan, "risks_to_check", []),
                "evidence_block": _format_evidence(evidence),
            })
            emitter and emitter.emit(
                "AgentFinished",
                agent="synthesizer",
                key_findings=len(brief.key_findings),
                risks=len(brief.risks),
            )
            return brief