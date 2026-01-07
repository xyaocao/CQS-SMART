from typing import Dict, Any, Tuple, List, Optional
import json
import re
import sys
import random
from pathlib import Path
import time
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.llm import LLMConfig, get_llm_chat_model
from baseline.planneragent import json_file, log_raw_response, get_response_text
from baseline.prompts_planner import Planner_system_prompt, Planner_human
from prompts_firstloop import Skeptic_system_prompt, Skeptic_human, Reasoner_system_prompt, Reasoner_human, SQLGen_system_prompt, SQLGen_human
from prompts_basemad_new import (Skeptic_basemad_system_prompt, Skeptic_basemad_human, Defender_basemad_system_prompt, Defender_basemad_human, 
                                Reasoner_basemad_system_prompt, Reasoner_basemad_human,  SQLGen_system_prompt_basemad,SQLGen_human_basemad)
from state import PlannerAgentState, SkepticAgentState, ReasonerAgentState, DefenderAgentState


def select_llm(loop_mode: str, model_pool: Optional[List], llm_pool: Optional[List[LLMConfig]], fixed_model, agent_name: str = "agent") -> Tuple:
    """Global function to select an LLM for an agent call.
    
    For heteroMAD mode, randomly selects from the model pool.
    For other modes, returns the fixed model.
    
    Args:
        loop_mode: The loop mode ("heteroMAD", "BaseMAD", "first")
        model_pool: List of instantiated ChatOpenAI models (for heteroMAD)
        llm_pool: List of LLMConfig objects (for heteroMAD)
        fixed_model: The fixed model to use for non-heteroMAD modes
        agent_name: Name of the agent (for logging/debugging)
    
    Returns:
        Tuple of (selected_model, model_name)
    """
    if loop_mode == "heteroMAD" and model_pool and llm_pool and len(model_pool) > 0:
        # Randomly select from the pool
        selected_idx = random.randint(0, len(model_pool) - 1)
        selected_model = model_pool[selected_idx]
        selected_config = llm_pool[selected_idx]
        model_name = selected_config.model
        return selected_model, model_name
    else:
        # Use fixed model for BaseMAD/firstloop modes
        model_name = getattr(fixed_model, 'model_name', 'default') if fixed_model else 'default'
        return fixed_model, model_name


class PlannerAgent:
    def __init__(self, llm_config: LLMConfig = None, llm_pool: Optional[List[LLMConfig]] = None, loop_mode: str = "first"):
        self.loop_mode = loop_mode
        self.llm_pool = llm_pool  # For heteroMAD: list of LLM configs to choose from
        if llm_pool and loop_mode == "heteroMAD":
            # heteroMAD: will select randomly from pool on each call
            self.model = None  # Will be selected dynamically
            self.model_pool = [get_llm_chat_model(cfg) for cfg in llm_pool]
        else:
            # BaseMAD/firstloop: use single fixed model
            self.model = get_llm_chat_model(llm_config)
            self.model_pool = None
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
            question=state.question,
            schema=state.schema_text,
        )
        model, model_name = select_llm(self.loop_mode, self.model_pool, self.llm_pool, self.model, "planner")
        if self.loop_mode == "heteroMAD":
            # Log which LLM was chosen
            try:
                log_raw_response("planner_llm_choice", model_name, state=state, also_print=False)
            except Exception:
                pass
        response = model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start
        content = get_response_text(response)
        log_raw_response("planner_repr", repr(response), state=state)
        log_raw_response("planner", content, state=state)
        state.plan = json_file(content)
        return state

    def run(self, question: str, schema_text: str) -> Tuple[Dict[str, Any], float]:
        state = PlannerAgentState(question=question, schema_text=schema_text or "")
        out_state = self.app.invoke(state)
        out = PlannerAgentState(**out_state)
        return out.plan, out.latency

class SkepticAgent:
    def __init__(self, llm_config: LLMConfig = None, critical_questions: str = "", loop_mode: str = "first", llm_pool: Optional[List[LLMConfig]] = None):
        self.loop_mode = loop_mode
        self.llm_pool = llm_pool  # For heteroMAD: list of LLM configs to choose from
        if llm_pool and loop_mode == "heteroMAD":
            # heteroMAD: will select randomly from pool on each call
            self.model = None  # Will be selected dynamically
            self.model_pool = [get_llm_chat_model(cfg) for cfg in llm_pool]
        else:
            # BaseMAD/firstloop: use single fixed model
            self.model = get_llm_chat_model(llm_config)
            self.model_pool = None
        # Choose prompts based on loop mode.
        if loop_mode in ("BaseMAD", "heteroMAD", "ImprovedMAD"):
            system_prompt = Skeptic_basemad_system_prompt
            human_prompt = Skeptic_basemad_human
        else:
            system_prompt = Skeptic_system_prompt
            human_prompt = Skeptic_human
        self.skeptic_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
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
            question=state.question,
            schema=state.schema_text,
            plan=state.plan,
            critical_questions=state.critical,
            execution_feedback=state.execution_feedback or "",
        )
        model, model_name = select_llm(self.loop_mode, self.model_pool, self.llm_pool, self.model, "skeptic")
        if self.loop_mode == "heteroMAD":
            # Log which LLM was chosen
            try:
                log_raw_response("skeptic_llm_choice", model_name, state=state, also_print=False)
            except Exception:
                pass
        response = model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start
        content = get_response_text(response)
        log_raw_response("skeptic_repr", repr(response), state=state)
        log_raw_response("skeptic", content, state=state)
        # If running BaseMAD or heteroMAD, detect common truncation/empty/unbalanced cases and
        # retry once with a short JSON-only instruction. This does not run for
        # first-loop so existing baseline/firstloop runs are unchanged.
        try:
            if self.loop_mode in ("BaseMAD", "heteroMAD", "ImprovedMAD"):
                repr_text = repr(response)
                unbalanced = (content.count("{") != content.count("}") or content.count("[") != content.count("]"))
                truncated_flag = ("finish_reason" in repr_text and "length" in repr_text)
                if (not content or not content.strip()) or unbalanced or truncated_flag:
                    try:
                        retry_msg = HumanMessage(content=(
                            "Your previous response appeared truncated or invalid. Reply ONLY with the compact JSON object requested and nothing else (no code fences, no explanation)."
                        ))
                        retry_messages = prompt_value.to_messages() + [retry_msg]
                        response2 = model.invoke(retry_messages)
                        content2 = get_response_text(response2)
                        # Use retry output if non-empty
                        if content2 and content2.strip():
                            content = content2
                            try:
                                log_raw_response("skeptic_retry_repr", repr(response2), state=state)
                            except Exception:
                                pass
                            try:
                                log_raw_response("skeptic_retry", content2, state=state)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

        state.feedback = json_file(content)
        return state
    
    def run(self, question: str, schema_text: str, plan: Dict[str, Any], execution_feedback: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], float]:
        state = SkepticAgentState(
            question=question,
            schema_text=schema_text or "",
            plan=json.dumps(plan, ensure_ascii=False),
            critical=self.critical_questions,
            execution_feedback=json.dumps(execution_feedback or {}, ensure_ascii=False),
        )
        out_state = self.app.invoke(state)
        out = SkepticAgentState(**out_state)
        return out.feedback, out.latency

class DefenderAgent:
    def __init__(self, llm_config: LLMConfig = None, loop_mode: str = "BaseMAD", llm_pool: Optional[List[LLMConfig]] = None):
        self.loop_mode = loop_mode
        self.llm_pool = llm_pool  # For heteroMAD: list of LLM configs to choose from
        if llm_pool and loop_mode == "heteroMAD":
            # heteroMAD: will select randomly from pool on each call
            self.model = None  # Will be selected dynamically
            self.model_pool = [get_llm_chat_model(cfg) for cfg in llm_pool]
        else:
            # BaseMAD: use single fixed model
            self.model = get_llm_chat_model(llm_config)
            self.model_pool = None
        # Defender is used in BaseMAD and heteroMAD
        system_prompt = Defender_basemad_system_prompt
        human_prompt = Defender_basemad_human
        self.defender_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        graph = StateGraph(DefenderAgentState)
        graph.add_node("defend", self.defender_node)
        graph.add_edge(START, "defend")
        graph.add_edge("defend", END)
        self.app = graph.compile()

    def defender_node(self, state: DefenderAgentState) -> DefenderAgentState:
        start = time.perf_counter()
        prompt_value = self.defender_prompt.format_prompt(
            question=state.question,
            schema=state.schema_text,
            plan=state.plan,
            skeptic_feedback=state.skeptic_feedback or "",
        )
        model, model_name = select_llm(self.loop_mode, self.model_pool, self.llm_pool, self.model, "defender")
        if self.loop_mode == "heteroMAD":
            # Log which LLM was chosen
            try:
                log_raw_response("defender_llm_choice", model_name, state=state, also_print=False)
            except Exception:
                pass
        response = model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start
        content = get_response_text(response)
        log_raw_response("defender_repr", repr(response), state=state)
        log_raw_response("defender", content, state=state)
        # Retry guard for BaseMAD and heteroMAD (defender is normally used in BaseMAD/heteroMAD)
        try:
            if self.loop_mode in ("BaseMAD", "heteroMAD", "ImprovedMAD"):
                repr_text = repr(response)
                unbalanced = (content.count("{") != content.count("}") or content.count("[") != content.count("]"))
                truncated_flag = ("finish_reason" in repr_text and "length" in repr_text)
                if (not content or not content.strip()) or unbalanced or truncated_flag:
                    try:
                        retry_msg = HumanMessage(content=(
                            "Your previous response appeared truncated or invalid. Reply ONLY with the compact JSON object requested and nothing else (no code fences, no explanation)."
                        ))
                        retry_messages = prompt_value.to_messages() + [retry_msg]
                        response2 = model.invoke(retry_messages)
                        content2 = get_response_text(response2)
                        if content2 and content2.strip():
                            content = content2
                            try:
                                log_raw_response("defender_retry_repr", repr(response2), state=state)
                            except Exception:
                                pass
                            try:
                                log_raw_response("defender_retry", content2, state=state)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

        state.feedback = json_file(content)
        return state

    def run(self, question: str, schema_text: str, plan: Dict[str, Any], skeptic_feedback: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], float]:
        state = DefenderAgentState(
            question=question,
            schema_text=schema_text or "",
            plan=json.dumps(plan, ensure_ascii=False),
            skeptic_feedback=json.dumps(skeptic_feedback or {}, ensure_ascii=False),
        )
        out_state = self.app.invoke(state)
        out = DefenderAgentState(**out_state)
        return out.feedback, out.latency


class ReasonerAgent:
    def __init__(self, llm_config: LLMConfig = None, loop_mode: str = "first", llm_pool: Optional[List[LLMConfig]] = None):
        self.loop_mode = loop_mode
        self.llm_pool = llm_pool  # For heteroMAD: list of LLM configs to choose from
        if llm_pool and loop_mode == "heteroMAD":
            # heteroMAD: will select randomly from pool on each call
            self.model = None  # Will be selected dynamically
            self.model_pool = [get_llm_chat_model(cfg) for cfg in llm_pool]
        else:
            # BaseMAD/firstloop: use single fixed model
            self.model = get_llm_chat_model(llm_config)
            self.model_pool = None
        # Choose decision prompts based on loop mode.
        if loop_mode in ("BaseMAD", "heteroMAD", "ImprovedMAD"):
            decision_system_prompt = Reasoner_basemad_system_prompt
            decision_human_prompt = Reasoner_basemad_human
        else:
            decision_system_prompt = Reasoner_system_prompt
            decision_human_prompt = Reasoner_human
        self.decision_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", decision_system_prompt),
                ("human", decision_human_prompt),
            ]
        )
      
        if loop_mode in ("BaseMAD", "heteroMAD", "ImprovedMAD"):
            sql_system_prompt = SQLGen_system_prompt_basemad
            sql_human_prompt = SQLGen_human_basemad
        else:
            sql_system_prompt = SQLGen_system_prompt
            sql_human_prompt = SQLGen_human
        self.sql_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sql_system_prompt),
                ("human", sql_human_prompt),
            ]
        )
        graph = StateGraph(ReasonerAgentState)
        graph.add_node("decision", self.decision_node)
        graph.add_node("sqlgen", self.sqlgen_node)
        graph.add_edge(START, "decision")
        graph.add_edge("decision", "sqlgen")
        graph.add_edge("sqlgen", END)
        self.app = graph.compile()

    @staticmethod
    def extract_schema_names(schema_text: str) -> Dict[str, set]:
        tables = set()
        columns = set()
        current = None
        for line in schema_text.splitlines():
            m = re.search(r"CREATE TABLE\s+\"?(\w+)\"?", line, re.IGNORECASE)
            if m:
                current = m.group(1).lower()
                tables.add(current)
                continue
            if current:
                cm = re.search(r"\"?(\w+)\"?\s", line)
                if cm:
                    col = cm.group(1).lower()
                    columns.add(f"{current}.{col}")
            if ")" in line:
                current = None
        return {"tables": tables, "columns": columns}

    @staticmethod
    def filter_adjustments(adjs: Any, schema_text: str) -> Any:
        if not isinstance(adjs, list):
            return adjs
        names = ReasonerAgent.extract_schema_names(schema_text or "")
        cols = names["columns"]
        kept = []
        for a in adjs:
            s = str(a)
            drop = False
            for tok in re.findall(r"(\w+\.\w+)", s):
                if tok.lower() not in cols:
                    drop = True
                    break
            if not drop:
                kept.append(a)
        return kept

    def decision_node(self, state: ReasonerAgentState) -> ReasonerAgentState:
        start = time.perf_counter()
        prompt_value = self.decision_prompt.format_prompt(
                question=state.question,
                schema=state.schema_text,
                plan=state.plan,
                feedback=state.feedback,
            )
        model, model_name = select_llm(self.loop_mode, self.model_pool, self.llm_pool, self.model, "reasoner_decision")
        if self.loop_mode == "heteroMAD":
            # Log which LLM was chosen
            try:
                log_raw_response("reasoner_decision_llm_choice", model_name, state=state, also_print=False)
            except Exception:
                pass
        response = model.invoke(prompt_value.to_messages())
        state.decision_latency = time.perf_counter() - start
        content = get_response_text(response)
        log_raw_response("reasoner_repr", repr(response), state=state)
        log_raw_response("reasoner", content, state=state)
        # Retry guard but only when running BaseMAD or heteroMAD (reasoner behavior in firstloop left unchanged)
        try:
            if self.loop_mode in ("BaseMAD", "heteroMAD", "ImprovedMAD"):
                repr_text = repr(response)
                unbalanced = (content.count("{") != content.count("}") or content.count("[") != content.count("]"))
                truncated_flag = ("finish_reason" in repr_text and "length" in repr_text)
                if (not content or not content.strip()) or unbalanced or truncated_flag:
                    try:
                        retry_msg = HumanMessage(content=(
                            "Your previous response appeared truncated or invalid. Reply ONLY with the compact JSON object requested and nothing else (no code fences, no explanation)."
                        ))
                        retry_messages = prompt_value.to_messages() + [retry_msg]
                        response2 = model.invoke(retry_messages)
                        content2 = get_response_text(response2)
                        if content2 and content2.strip():
                            content = content2
                            try:
                                log_raw_response("reasoner_retry_repr", repr(response2), state=state)
                            except Exception:
                                pass
                            try:
                                log_raw_response("reasoner_retry", content2, state=state)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

        state.decision = json_file(content)
        # Guard: if skeptic verdict was block/warn, force revise
        try:
            fb = json.loads(state.feedback or "{}")
            skept_verdict = str(fb.get("final_skeptic_verdict", "")).lower()
            if skept_verdict in {"block", "warn"} and isinstance(state.decision, dict):
                state.decision["verdict"] = "revise"
        except Exception:
            pass
        return state

    def sqlgen_node(self, state: ReasonerAgentState) -> ReasonerAgentState:
        start = time.perf_counter()
        if isinstance(state.decision, dict) and "adjustments" in state.decision:
            state.decision["adjustments"] = self.filter_adjustments(
                state.decision.get("adjustments", []),
                state.schema_text,
            )
        prompt_value = self.sql_prompt.format_prompt(
            question=state.question,
            schema=state.schema_text,
            plan=state.plan,
            decision=json.dumps(state.decision or {}, ensure_ascii=False),
        )
        model, model_name = select_llm(self.loop_mode, self.model_pool, self.llm_pool, self.model, "reasoner_sqlgen")
        if self.loop_mode == "heteroMAD":
            # Log which LLM was chosen
            try:
                log_raw_response("reasoner_sqlgen_llm_choice", model_name, state=state, also_print=False)
            except Exception:
                pass
        response = model.invoke(prompt_value.to_messages())
        state.sql_latency = time.perf_counter() - start
        sql_text = get_response_text(response)
        log_raw_response("sqlgen_repr", repr(response), state=state)
        log_raw_response("sqlgen", sql_text, state=state)
        state.sql = self.strip_reasoning(sql_text)
        return state
    
    def run(self, question: str, schema_text: str, plan: Dict[str, Any], feedback: Dict[str, Any], exec_feedback: Dict[str, Any] | None = None,) -> Tuple[Dict[str, Any], float, str, float]:
        state = ReasonerAgentState(
            question=question,
            schema_text=schema_text or "",
            plan=json.dumps(plan, ensure_ascii=False),
            feedback=json.dumps(feedback, ensure_ascii=False),
            execution_feedback=json.dumps(exec_feedback or {}, ensure_ascii=False),
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


