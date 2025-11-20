from typing import Dict, Any, Tuple
import json
import re
import sys
from pathlib import Path
import time
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.llm import LLMConfig, get_llm_chat_model
from baseline.planneragent import json_file
from baseline.prompts_planner import Planner_system_prompt, Planner_human
from prompts_firstloop import (Skeptic_system_prompt, Skeptic_human, Reasoner_system_prompt, Reasoner_human, SQLGen_system_prompt, SQLGen_human)
from state import PlannerAgentState, SkepticAgentState, ReasonerAgentState

class PlannerAgent:
    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Planner_system_prompt),
                ("human", Planner_human),
            ]
        )
        graph = StateGraph(PlannerAgentState)
        graph.add_node("plan", self.planner_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", END)
        self.app = graph.compile()

    def planner_node(self, state: PlannerAgentState) -> PlannerAgentState:
        start = time.perf_counter() 
        prompt_value = self.plan_prompt.format_prompt(
                question = state.question,
                schema = state.schema_text or "",
            )
        response = self.model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start
        content = response.content if hasattr(response, "content") else str(response)
        state.plan = json_file(content)
        # state.raw = content
        return state 

    def run(self, question: str, schema_text: str) -> Tuple[Dict[str, Any], float, str]:
        state = PlannerAgentState(question=question, schema_text=schema_text or "")
        out_state = self.app.invoke(state)
        out = PlannerAgentState(**out_state) 
        return out.plan, out.latency

class SkepticAgent:
    def __init__(self, llm_config: LLMConfig = None, critical_questions: str = ""):
        self.model = get_llm_chat_model(llm_config)
        self.skeptic_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Skeptic_system_prompt),
                ("human", Skeptic_human),
            ]
        )
        self.critical_questions = critical_questions.strip() or "N/A"
        graph = StateGraph(SkepticAgentState)
        graph.add_node("critique", self.skeptic_node)
        graph.add_edge(START, "critique")
        graph.add_edge("critique", END)
        self.app = graph.compile()

    def skeptic_node(self, state: SkepticAgentState) -> SkepticAgentState:
        start = time.perf_counter()
        prompt_value = self.skeptic_prompt.format_prompt(
            question = state.question,
            schema = state.schema_text or "",
            plan = state.plan,
            critical_questions = state.critical,
        )
        response = self.model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start
        content = response.content if hasattr(response, "content") else str(response)
        state.feedback = json_file(content)
        # state.raw = content
        return state
    
    def run(self, question: str, schema_text: str, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], float, str]:
        state = SkepticAgentState(
            question = question,
            schema_text = schema_text or "",
            plan = json.dumps(plan, ensure_ascii=False),
            critical = self.critical_questions,
        )
        out_state = self.app.invoke(state)
        out = SkepticAgentState(**out_state)
        return out.feedback, out.latency
    
class ReasonerAgent:
    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.decision_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", Reasoner_system_prompt),
                ("human", Reasoner_human),
            ]
        )
        self.sql_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SQLGen_system_prompt),
                ("human", SQLGen_human),
            ]
        )
        graph = StateGraph(ReasonerAgentState)
        graph.add_node("decision", self.decision_node)
        graph.add_node("sqlgen", self.sqlgen_node)
        graph.add_edge(START, "decision")
        graph.add_edge("decision", "sqlgen")
        graph.add_edge("sqlgen", END)
        self.app = graph.compile()

    def decision_node(self, state: ReasonerAgentState) -> ReasonerAgentState:
        start = time.perf_counter()
        prompt_value = self.decision_prompt.format_prompt(
            question = state.question,
            schema = state.schema_text or "",
            plan = state.plan,
            feedback = state.feedback,
        )
        response = self.model.invoke(prompt_value.to_messages())
        state.decision_latency = time.perf_counter() - start
        content = response.content if hasattr(response, "content") else str(response)
        state.decision = json_file(content)
        # state.decision_raw = content
        return state

    def sqlgen_node(self, state: ReasonerAgentState) -> ReasonerAgentState:
        start = time.perf_counter()
        prompt_value = self.sql_prompt.format_prompt(
            question = state.question,
            schema = state.schema_text or "",
            decision = json.dumps(state.decision or {}, ensure_ascii=False,),
        )
        response = self.model.invoke(prompt_value.to_messages())
        state.sql_latency = time.perf_counter() - start
        sql_text = response.content if hasattr(response, "content") else str(response)
        state.sql = self.strip_reasoning(sql_text)
        # state.sql_raw = sql_text
        return state
    
    def run(self, question: str, schema_text: str, plan: Dict[str, Any], feedback: Dict[str, Any]) -> Tuple[Dict[str, Any], float, str, str, float, str]:
        state = ReasonerAgentState(
            question = question,
            schema_text = schema_text or "",
            plan = json.dumps(plan, ensure_ascii=False),
            feedback = json.dumps(feedback, ensure_ascii=False),
        )
        out_state = self.app.invoke(state)
        out = ReasonerAgentState(**out_state)
        return out.decision, out.decision_latency, out.sql, out.sql_latency  

    @staticmethod
    def strip_reasoning(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        lower = cleaned.lower()
        if "```" in cleaned:
            if "```sql" in lower:
                cleaned = cleaned[lower.find("```sql") + 6:]
            else:
                cleaned = cleaned[cleaned.find("```") + 3:]
            if "```" in cleaned:
                cleaned = cleaned[: cleaned.find("```")]
        return cleaned.strip()   

