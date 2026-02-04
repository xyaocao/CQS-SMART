from typing import Dict, Any
import json
import ast
import re
import sys 
import time
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.state import PlannerState
from baseline.prompts_planner import Planner_system_prompt, Planner_human, SQLGen_system_prompt, SQLGen_human
from baseline.llm import get_llm_chat_model, LLMConfig
from baseline.parse import json_file, get_response_text, log_raw_response, extract_sql

class PlannerGraph:
    """Graph for the baseline planner agent."""
    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.plan_prompt = ChatPromptTemplate.from_messages([
            ("system", Planner_system_prompt),
            ("human", Planner_human),
        ])
        self.sqlgen_prompt = ChatPromptTemplate.from_messages([
            ("system", SQLGen_system_prompt),
            ("human", SQLGen_human),
        ]) 

        graph = StateGraph(PlannerState)
        graph.add_node("plan", self.planner_node)
        graph.add_node("sqlgen", self.sqlgen_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "sqlgen")
        graph.add_edge("sqlgen", END)
        self.app = graph.compile()       

    def planner_node(self, state: PlannerState) -> PlannerState:
            prompt_value = self.plan_prompt.format_prompt(
                question = state.question,
                schema = state.schema_text or "",
            )
            response = self.model.invoke(prompt_value.to_messages())
            # Extract text robustly from the response object and also record the object's repr
            try:
                raw = get_response_text(response)
            except Exception:
                raw = ""
            try:
                log_raw_response("planner_repr", repr(response), state=state)
            except Exception:
                pass
            # Save and print the extracted textual content
            log_raw_response("planner", raw, state=state)

            # If the model returned an empty/whitespace response, retry once with a short
            # instruction asking for JSON-only. This keeps behavior for Qwen the same
            # while helping recover from silent empties from other models.
            try:
                if not raw or not raw.strip():
                    try:
                        retry_msg = HumanMessage(content=(
                            "Your previous response was empty. Reply ONLY with the JSON object requested,"
                            " and nothing else (no explanation, no code fences)."
                        ))
                        retry_messages = prompt_value.to_messages() + [retry_msg]
                        response2 = self.model.invoke(retry_messages)
                        try:
                            raw2 = get_response_text(response2)
                        except Exception:
                            raw2 = ""
                        try:
                            log_raw_response("planner_retry_repr", repr(response2), state=state)
                        except Exception:
                            pass
                        log_raw_response("planner_retry", raw2, state=state)
                        # Use retry output if non-empty
                        if raw2 and raw2.strip():
                            raw = raw2
                    except Exception:
                        # If retry fails, continue to attempt parsing the original (empty) raw
                        pass
            except Exception:
                pass

            plan = json_file(raw)
            state.plan = plan
            return state

    def sqlgen_node(self, state: PlannerState) -> PlannerState:
            prompt_value = self.sqlgen_prompt.format_prompt(
                question = state.question,
                schema = state.schema_text or "",
                plan = json.dumps(state.plan, ensure_ascii=False)
            )
            response = self.model.invoke(prompt_value.to_messages())
            try:
                sql_text = get_response_text(response)
            except Exception:
                sql_text = ""
            try:
                log_raw_response("sqlgen_repr", repr(response), state=state)
            except Exception:
                pass
            # Also log raw SQL-generation LLM output for inspection and model adaptation.
            log_raw_response("sqlgen", sql_text, state=state)
            state.sql = extract_sql(sql_text)
            return state
    
    def invoke(self, state: PlannerState) -> PlannerState:
        return self.app.invoke(state)