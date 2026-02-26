import json
from typing import Optional
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel

from engine.schemas.planner import ResearchPlan
from engine.events.emitter import Emitter


PLANNER_SYSTEM = """You are a senior strategy analyst and research planner.
Your job: turn a vague business question into a research plan that can produce a concise, defensible brief.
Return ONLY valid JSON that matches the ResearchPlan schema.
Do not use markdown. Do not wrap JSON in <function=...> tags. Do not add commentary.

Rules:
- Subquestions must cover: market, users/buyers, competitors/alternatives, economics/pricing, risks (regulatory/operational), and execution considerations.
- Search queries must be specific and diverse (mix market sizing, competitor docs, analyst reports, regulatory sources, and credible news).
- Keep it tool-friendly: queries should be copy/paste ready.
- Include assumptions explicitly if the question is underspecified.
- Assumption must be a list of objects - assumption: str, rationale: Optional[str].
- Keep scope realistic: do not propose more work than needed.
"""

PLANNER_HUMAN = """Question: {question}

Constraints:
- Budget (USD est): {budget_usd}
- Time limit (seconds): {time_limit_s}

Output JSON only.
"""

def _to_research_plan(msg) -> ResearchPlan:
    txt = msg.content.strip()
    data = json.loads(txt)
    return ResearchPlan.model_validate(data)

def build_planner_runnable(llm: BaseChatModel) -> Runnable:
    """
    Returns a runnable that maps {question, budget_usd, time_limit_s} -> ResearchPlan
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", PLANNER_SYSTEM), ("human", PLANNER_HUMAN)]
    )

    # Ask the model for JSON object output when supported
    llm_json = llm.bind(response_format={"type": "json_object"})

    return prompt | llm_json | RunnableLambda(_to_research_plan)

@dataclass
class PlannerAgent:
    llm: BaseChatModel

    def plan(
            self, 
            question: str, 
            budget_usd: float, 
            time_limit_s: Optional[int] = None,
            emitter: Optional[Emitter] = None,
        ) -> ResearchPlan:
        
        emitter and emitter.emit("AgentStarted", agent="planner", question=question, budget_usd=budget_usd, time_limit_s=time_limit_s)

        runnable = build_planner_runnable(self.llm)
        plan = runnable.invoke(
            {
                "question": question,
                "budget_usd": budget_usd,
                "time_limit_s": time_limit_s or 0,
            }
        )

        emitter and emitter.emit(
            "PlanCreated",
            agent="planner",
            subquestions_count=len(plan.subquestions),
            search_queries_count=len(plan.search_queries),
            assumptions_count=len(plan.assumptions),
        )
        emitter and emitter.emit("AgentFinished", agent="planner")
        return plan