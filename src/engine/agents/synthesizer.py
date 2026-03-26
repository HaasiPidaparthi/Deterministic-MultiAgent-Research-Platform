"""Synthesis agent for generating executive briefs from research evidence.

This module contains the SynthesizerAgent class, which is responsible for synthesizing
research evidence into structured executive briefs. The agent uses LLM prompts to
generate comprehensive briefs with key findings, risks, recommendations, and other
structured elements based on the research question, plan, and collected evidence.
"""

import json
from dataclasses import dataclass
from typing import List, Optional

from pydantic import ValidationError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from engine.schemas.brief import BriefDraft, Claim
from engine.schemas.evidence import EvidenceItem
from engine.schemas.planner import ResearchPlan
from engine.events.emitter import Emitter
from engine.metrics.llm_usage import add_llm_usage


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
- Set confidence to 0.0 for all claims.
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

def _to_brief(msg, evidence: Optional[List[EvidenceItem]] = None) -> BriefDraft:
    txt = msg.content.strip()
    data = json.loads(txt)
    try:
        return BriefDraft.model_validate(data)
    except ValidationError as exc:
        errs = exc.errors()
        key_findings_issue = any(
            err.get("type") == "too_short" and tuple(err.get("loc", [])) == ("key_findings",)
            for err in errs
        )
        if not key_findings_issue:
            raise

        existing = data.get("key_findings", []) or []
        while len(existing) < 3:
            existing.append(
                {
                    "text": "Insufficient key findings from model output; placeholder claim inserted.",
                    "citations": [evidence[0].id] if evidence and len(evidence) >= 1 else [],
                    "confidence": 0.0,
                }
            )
        data["key_findings"] = existing
        return BriefDraft.model_validate(data)


def _ensure_min_key_findings(brief: BriefDraft, evidence: List[EvidenceItem]) -> BriefDraft:
    if len(brief.key_findings) >= 3:
        return brief

    placeholder_id = evidence[0].id if evidence else "S0"
    existing = list(brief.key_findings)
    while len(existing) < 3:
        existing.append(
            Claim(
                text="Insufficient key findings from model output; placeholder claim inserted.",
                citations=[placeholder_id] if placeholder_id else [],
                confidence=0.0,
            )
        )

    brief.key_findings = existing
    return brief


@dataclass
class SynthesizerAgent:
    """Agent responsible for synthesizing research evidence into executive briefs.

    This agent takes a research question, plan, and collected evidence items,
    then uses an LLM to generate a structured executive brief containing
    key findings, risks, recommendations, and other analysis elements.

    Attributes:
        llm: The language model used for synthesis and brief generation.
    """
    llm: BaseChatModel

    def synthesize(
        self,
        question: str,
        plan: ResearchPlan,
        evidence: List[EvidenceItem],
        emitter: Optional[Emitter] = None,
        mode: str = "normal",
        metrics: Optional[dict] = None,
    ) -> BriefDraft:
        """Synthesize research evidence into a structured executive brief.

        Uses the provided research question, plan, and evidence items to generate
        a comprehensive executive brief with key findings, risks, recommendations,
        and other structured elements.

        Args:
            question: The main research question being addressed.
            plan: The research plan containing subquestions, assumptions, and risks to check.
            evidence: List of evidence items collected during research.
            emitter: Optional event emitter for logging agent progress and events.
            mode: Synthesis mode - "normal" or "strict" for tighter requirements.
            metrics: Optional dictionary to track LLM usage and performance metrics.

        Returns:
            A BriefDraft object containing the synthesized executive brief with
            title, executive summary, key findings, risks, recommendations, etc.

        Raises:
            Exception: If synthesis fails, returns a BriefDraft with error information.
        """
        emitter and emitter.emit("AgentStarted", agent="synthesizer", question=question, evidence_count=len(evidence))

        # Build inputs once
        subq = [getattr(s, "question", str(s)) for s in getattr(plan, "subquestions", [])]
        assumptions = [getattr(a, "assumption", str(a)) for a in getattr(plan, "assumptions", [])]
        payload = {
            "question": question,
            "subquestions": subq,
            "assumptions": assumptions,
            "risks_to_check": getattr(plan, "risks_to_check", []),
            "evidence_block": _format_evidence(evidence),
        }

        def _model_name() -> str:
            return getattr(self, "model_name", None) or getattr(self.llm, "model", None) or "unknown"

        # Mode to tighten requirements
        system = SYNTH_SYSTEM
        if mode == "strict":
            system = system + "\n\n" + REPAIR_SYSTEM

        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", SYNTH_HUMAN)])

        # JSON-only output
        llm_json = self.llm.bind(response_format={"type": "json_object"}, temperature=0)

        try:
            # 1) run model -> AIMessage
            msg = (prompt | llm_json).invoke(payload)

            # 2) account tokens/cost
            if metrics is not None:
                add_llm_usage(metrics, msg, _model_name())

            # 3) parse into BriefDraft
            brief = _to_brief(msg, evidence)

            # 4) enforce minimum findings
            if len(brief.key_findings) < 3:
                brief = _ensure_min_key_findings(brief, evidence)
                emitter and emitter.emit(
                    "AgentWarning",
                    agent="synthesizer",
                    issue="insufficient_key_findings",
                    key_findings=len(brief.key_findings),
                )

            emitter and emitter.emit(
                "AgentFinished",
                agent="synthesizer",
                key_findings=len(brief.key_findings),
                risks=len(brief.risks),
            )
            return brief
        except Exception as e:
            # If parsing fails, add placeholders
            brief = BriefDraft(
                title="Error in Synthesis",
                executive_summary="An error occurred during synthesis.",
                key_findings=[
                    Claim(
                        text="Synthesis failed due to error.",
                        citations=[],
                        confidence=0.0,
                    )
                ] * 3,
                risks=[],
                recommendation="Retry the synthesis.",
                next_steps=[],
                assumptions=[],
                limitations=["Synthesis error: " + str(e)],
            )
            emitter and emitter.emit(
                "AgentError",
                agent="synthesizer",
                error=str(e),
            )
            return brief