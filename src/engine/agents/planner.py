"""
Planner Agent Module

This module contains the PlannerAgent class which is responsible for breaking down
business questions into structured research plans. The planner analyzes the input
question and generates subquestions, search queries, assumptions, and risk areas
to guide the research process.
"""

import json
from typing import Optional
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser

from engine.schemas.planner import ResearchPlan
from engine.events.emitter import Emitter
from engine.metrics.llm_usage import extract_token_usage, add_llm_usage


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

def build_planner_chain(llm: BaseChatModel):
    """
    Returns a runnable that maps {question, budget_usd, time_limit_s} -> ResearchPlan
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", PLANNER_SYSTEM), ("human", PLANNER_HUMAN)]
    )
    llm_step = prompt | llm  # returns AIMessage
    parser = JsonOutputParser(pydantic_object=ResearchPlan)

    return llm_step, parser

@dataclass
class PlannerAgent:
    """
    Agent responsible for planning research activities.

    The PlannerAgent takes a business question and generates a structured research plan
    that includes subquestions, search queries, assumptions, and risk areas to investigate.
    This plan guides the subsequent research and analysis phases.
    """
    llm: BaseChatModel

    def plan(
            self, 
            question: str, 
            budget_usd: float, 
            time_limit_s: Optional[int] = None,
            emitter: Optional[Emitter] = None,
            metrics: Optional[dict] = None,
        ) -> ResearchPlan:
        """
        Generate a structured research plan for the given question.

        Args:
            question: The business question to research
            budget_usd: Estimated budget in USD for the research
            time_limit_s: Optional time limit in seconds
            emitter: Optional event emitter for logging
            metrics: Optional metrics dictionary for tracking LLM usage

        Returns:
            ResearchPlan: Structured plan with subquestions, search queries, assumptions, and risks
        """
        
        emitter and emitter.emit("AgentStarted", agent="planner", question=question, budget_usd=budget_usd, time_limit_s=time_limit_s)

        llm_step, parser = build_planner_chain(self.llm)
        plan = llm_step.invoke(
            {
                "question": question,
                "budget_usd": budget_usd,
                "time_limit_s": time_limit_s or 0,
            }
        )

        # meter tokens/cost
        if metrics is not None:
            model_name = getattr(self, "model_name", None) or getattr(self.llm, "model", None)
            add_llm_usage(metrics, model_name, plan)                                      

        # parse to ResearchPlan
        plan = parser.invoke(plan)
        if isinstance(plan, dict):
            plan = ResearchPlan.model_validate(plan)
        elif not isinstance(plan, ResearchPlan):
            plan = ResearchPlan.model_validate(plan)

        emitter and emitter.emit(
            "PlanCreated",
            agent="planner",
            subquestions_count=len(plan.subquestions),
            search_queries_count=len(plan.search_queries),
            assumptions_count=len(plan.assumptions),
        )
        emitter and emitter.emit("AgentFinished", agent="planner")
        return plan