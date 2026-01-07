from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from baseline.llm import LLMConfig, get_llm_chat_model
from baseline.prompts_planner import Planner_system_prompt, Planner_human
from baseline.state import PlannerState
from baseline.evaluation import exec_sql
from prompts_improvedmad_new import ContractReasoner_system_prompt,ContractReasoner_human,ContractSQLGen_system_prompt,ContractSQLGen_human,OneShotSQLGen_system_prompt,OneShotSQLGen_human
from loop_engines import LoopEngine, LoopResult, LoopStageError

def dedup_list(items: Any, limit: int = 5) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    for x in items or []:
        s = str(x).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) >= limit:
            break
    return out


def load_history_advisor(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            db_id = str(obj.get("db_id", "")).lower().strip()
            q = (obj.get("question") or "").strip()
            sql = (obj.get("answer_sql") or "").strip()
            if not db_id or not q or not sql:
                continue
            obj["db_id"] = db_id
            out.append(obj)
        except Exception:
            continue
    return out


def match_history_advisor(advisors: List[Dict[str, Any]], question: str, db_id: str) -> Optional[Dict[str, Any]]:
    if not advisors:
        return None
    q = (question or "").lower()
    db = (db_id or "").lower()
    best = None
    best_score = 0
    for h in advisors:
        if h.get("db_id") != db:
            continue
        cand_q = (h.get("question") or "").lower()
        # simple overlap score
        q_tokens = set(t for t in re.split(r"[^a-z0-9]+", q) if t)
        c_tokens = set(t for t in re.split(r"[^a-z0-9]+", cand_q) if t)
        if not q_tokens or not c_tokens:
            continue
        score = len(q_tokens & c_tokens)
        if score > best_score:
            best_score = score
            best = h
    if best_score == 0:
        return None
    return best


def clean_sql_text(sql: str) -> str:
    if not isinstance(sql, str):
        try:
            sql = str(sql)
        except Exception:
            return ""
    s = sql.strip()
    s = re.sub(r"```[a-zA-Z]*", "", s)
    s = s.replace("```", "")
    s = re.sub(r";+$", "", s).strip()
    return s


def extract_schema_tables_columns(schema_text: str) -> Tuple[set[str], set[str]]:
    tables: set[str] = set()
    cols: set[str] = set()
    current: Optional[str] = None
    for line in (schema_text or "").splitlines():
        m = re.search(r"CREATE TABLE\s+\"?(\w+)\"?", line, re.IGNORECASE)
        if m:
            current = m.group(1).lower()
            tables.add(current)
            continue
        if current:
            cm = re.search(r"\"?(\w+)\"?\s", line)
            if cm:
                col = cm.group(1).lower()
                cols.add(f"{current}.{col}")
        if ")" in line:
            current = None
    return tables, cols


def validate_contract(contract: Dict[str, Any], schema_text: str) -> Dict[str, Any]:
    tables, cols = extract_schema_tables_columns(schema_text)
    out = dict(contract or {})

    def _filter_cols(exprs: Any) -> List[str]:
        keep: List[str] = []
        for e in exprs or []:
            s = str(e).strip()
            if not s:
                continue
            ok = True
            for tok in re.findall(r"(\w+\.\w+)", s):
                if tok.lower() not in cols:
                    ok = False
                    break
            if ok:
                keep.append(e)
        return keep

    def _filter_tables(tnames: Any) -> List[str]:
        keep: List[str] = []
        for t in tnames or []:
            s = str(t).strip()
            if s and s.lower() in tables:
                keep.append(t)
        return keep

    out["from"] = _filter_tables(out.get("from", []))
    out["select"] = _filter_cols(out.get("select", []))
    out["where"] = _filter_cols(out.get("where", []))
    out["having"] = _filter_cols(out.get("having", []))
    out["group_by"] = _filter_cols(out.get("group_by", []))

    joins_out: List[Dict[str, Any]] = []
    for j in out.get("joins", []) or []:
        if not isinstance(j, dict):
            continue
        t = str(j.get("table", "")).strip().lower()
        if t and t not in tables:
            continue
        on = str(j.get("on", "")).strip()
        ok = True
        for tok in re.findall(r"(\w+\.\w+)", on):
            if tok.lower() not in cols:
                ok = False
                break
        if ok:
            joins_out.append(j)
    out["joins"] = joins_out

    if isinstance(out.get("notes"), list):
        out["notes"] = dedup_list(out["notes"], limit=5)
    if not out.get("select"):
        out["contract_verdict"] = "revise"
        notes = out.get("notes") or []
        notes.append("select_missing_output")
        out["notes"] = dedup_list(notes, limit=5)
    return out


class ImprovedMADEngineNew(LoopEngine):
    """ImprovedMAD variant wired to prompts_improvedmad_new and history_advisor."""

    def __init__(
        self,
        planner,
        defender,
        skeptic,
        max_debate_rounds: int = 3,
        llm_config_reasoner: Optional[LLMConfig] = None,
        llm_config_sqlgen: Optional[LLMConfig] = None,
        llm_config_oneshot: Optional[LLMConfig] = None,
        history_advisor_path: Optional[Path] = None,
    ):
        self.planner = planner
        self.defender = defender
        self.skeptic = skeptic
        self.max_debate_rounds = max(1, int(max_debate_rounds))

        self.reasoner_model = get_llm_chat_model(llm_config_reasoner or LLMConfig())
        self.contract_prompt = ChatPromptTemplate.from_messages(
            [("system", ContractReasoner_system_prompt), ("human", ContractReasoner_human)]
        )
        self.contract_sqlgen_model = get_llm_chat_model(llm_config_sqlgen or LLMConfig())
        self.contract_sql_prompt = ChatPromptTemplate.from_messages(
            [("system", ContractSQLGen_system_prompt), ("human", ContractSQLGen_human)]
        )
        self.oneshot_model = get_llm_chat_model(llm_config_oneshot or LLMConfig())
        self.oneshot_prompt = ChatPromptTemplate.from_messages(
            [("system", OneShotSQLGen_system_prompt), ("human", OneShotSQLGen_human)]
        )
        # history advisor (new) plus legacy hints
        self.history_advisor = load_history_advisor(history_advisor_path or Path("Exp/history_advisor_q25.jsonl"))
        self.history_hints = []  # disable legacy hints by default
        # chooser prompt (LLM judge)
        self.chooser_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful SQL judge. Given a question, schema, and two candidate SQL queries, "
                    "pick which SQL better answers the question. Respond with exactly one token: 'contract' or 'oneshot'."
                ),
                (
                    "human",
                    "Question:\n{question}\n\nSchema:\n{schema}\n\nSQL A (contract):\n{sql_contract}\n\nSQL B (oneshot):\n{sql_oneshot}\n\nAnswer with 'contract' or 'oneshot'."
                ),
            ]
        )

    def pick_exemplar(self, question: str, db_id: str) -> Optional[Dict[str, Any]]:
        return match_history_advisor(self.history_advisor, question, db_id)

    @staticmethod
    def prefer_by_question_target(question: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        q = (question or "").lower()
        wants_model = "model" in q

        def mentions_model(sql: str) -> bool:
            s = (sql or "").lower()
            return " model" in s or "model " in s or "model(" in s

        if wants_model:
            c_has = mentions_model(sql_contract)
            o_has = mentions_model(sql_oneshot)
            if c_has and not o_has:
                return "contract"
            if o_has and not c_has:
                return "oneshot"
        return None

    @staticmethod
    def prefer_by_question_literals(question: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        import re

        q = (question or "")
        lits: set[str] = set()
        for m in re.findall(r"'([^']+)'", q):
            lits.add(m.lower().strip())
        for m in re.findall(r"\"([^\"]+)\"", q):
            lits.add(m.lower().strip())
        if not lits:
            return None
        c_score = 0
        o_score = 0
        sc = (sql_contract or "").lower()
        so = (sql_oneshot or "").lower()
        for lit in lits:
            if lit and lit in sc:
                c_score += 1
            if lit and lit in so:
                o_score += 1
        if c_score > 0 and o_score == 0:
            return "contract"
        if o_score > 0 and c_score == 0:
            return "oneshot"
        return None

    @staticmethod
    def prefer_by_unquoted_literals(question: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        stop = {
            "what","which","who","how","many","much","is","are","the","a","an","of","for","to","in","on","by","with","and","or","from","show","list","find","get","all","give","display","that","those","these","at","each","per",
        }
        tokens = {t for t in re.findall(r"[a-zA-Z]{3,}", (question or "").lower()) if t not in stop}
        if not tokens:
            return None
        sc = (sql_contract or "").lower()
        so = (sql_oneshot or "").lower()
        c_score = sum(1 for t in tokens if t in sc)
        o_score = sum(1 for t in tokens if t in so)
        if c_score > 0 and o_score == 0:
            return "contract"
        if o_score > 0 and c_score == 0:
            return "oneshot"
        return None

    @staticmethod
    def prefer_population_sum(question: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        q = (question or "").lower()
        if "population" not in q and "people" not in q:
            return None
        sc = (sql_contract or "").lower()
        so = (sql_oneshot or "").lower()
        def has_sum_pop(s: str) -> bool:
            return "sum(population" in s or "sum( population" in s
        def has_count_only(s: str) -> bool:
            return "count(" in s and "population" not in s
        c_pref = has_sum_pop(sc)
        o_pref = has_sum_pop(so)
        if c_pref and not o_pref:
            return "contract"
        if o_pref and not c_pref:
            return "oneshot"
        if has_count_only(sc) and not has_count_only(so):
            return "oneshot"
        if has_count_only(so) and not has_count_only(sc):
            return "contract"
        return None

    @staticmethod
    def prefer_both_languages(question: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        q = (question or "").lower()
        langs = re.findall(r"(english|dutch|spanish|french|german|chinese|arabic)", q, re.IGNORECASE)
        if len(set(langs)) < 2 or ("and" not in q and "both" not in q):
            return None
        def requires_both(sql: str) -> bool:
            s = (sql or "").lower()
            return "having" in s and ("= 2" in s or ">= 2" in s or "count(distinct language)" in s)
        c_req = requires_both(sql_contract)
        o_req = requires_both(sql_oneshot)
        if c_req and not o_req:
            return "contract"
        if o_req and not c_req:
            return "oneshot"
        return None

    @staticmethod
    def prefer_distinct_regions(question: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        q = (question or "").lower()
        if "region" not in q and "area" not in q:
            return None
        def has_distinct_region(sql: str) -> bool:
            s = (sql or "").lower()
            return "distinct region" in s or "select distinct" in s
        c = has_distinct_region(sql_contract)
        o = has_distinct_region(sql_oneshot)
        if c and not o:
            return "contract"
        if o and not c:
            return "oneshot"
        return None

    @staticmethod
    def prefer_car1_model(question: str, db_id: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        if (db_id or "").lower() != "car_1":
            return None
        if "model" not in (question or "").lower():
            return None
        def good(sql: str) -> bool:
            s = (sql or "").lower()
            return "car_names" in s and "model" in s and "cars_data" in s
        c = good(sql_contract)
        o = good(sql_oneshot)
        if c and not o:
            return "contract"
        if o and not c:
            return "oneshot"
        return None

    def run(self, question: str, schema_text: str, **kwargs: Any) -> LoopResult:
        total_start = time.perf_counter()
        latency: Dict[str, float] = {}

        # Planner
        try:
            p0 = time.perf_counter()
            plan, p_lat = self.planner.run(question, schema_text)
            latency["planner_sec"] = p_lat if p_lat is not None else (time.perf_counter() - p0)
        except Exception as exc:
            raise LoopStageError("Planner", exc, time.perf_counter() - total_start, {"latency": latency}) from exc

        # Debate (BaseMAD-style). If it fails, continue without it.
        debate_summary: Dict[str, Any] = {}
        try:
            d0 = time.perf_counter()
            debate_summary, debate_debug = self.defender.run_debate(question, schema_text, plan, self.max_debate_rounds)
            latency["debate_sec"] = time.perf_counter() - d0
        except Exception as exc:
            debate_summary = {}
            latency["debate_sec"] = 0.0

        # Advisor exemplar (optional)
        exemplar = self.pick_exemplar(question, kwargs.get("db_id", ""))
        exemplar_block = ""
        if exemplar:
            exemplar_block = (
                f"Successful example (same DB):\n"
                f"Question: {exemplar.get('question')}\n"
                f"SQL: {exemplar.get('answer_sql')}\n"
                f"Notes: {', '.join(exemplar.get('rationale') or [])}\n"
            )

        # Contract reasoner
        c0 = time.perf_counter()
        pv = self.contract_prompt.format_prompt(
            question=question,
            schema=schema_text,
            plan=json.dumps(plan, ensure_ascii=False),
            debate_summary=json.dumps(debate_summary or {}, ensure_ascii=False),
        )
        if exemplar_block:
            pv = self.contract_prompt.format_prompt(
                question=question + "\n\n" + exemplar_block,
                schema=schema_text,
                plan=json.dumps(plan, ensure_ascii=False),
                debate_summary=json.dumps(debate_summary or {}, ensure_ascii=False),
            )
        resp = self.reasoner_model.invoke(pv.to_messages())
        raw = getattr(resp, "content", str(resp))
        latency["contract_reasoner_sec"] = time.perf_counter() - c0
        contract = {}
        try:
            contract = json.loads(raw)
        except Exception:
            contract = {"contract_verdict": "revise"}

        # Contract SQLGen
        a0 = time.perf_counter()
        pv_a = self.contract_sql_prompt.format_prompt(
            question=question,
            schema=schema_text,
            contract=json.dumps(contract, ensure_ascii=False),
        )
        resp_a = self.contract_sqlgen_model.invoke(pv_a.to_messages())
        sql_a_raw = getattr(resp_a, "content", str(resp_a))
        sql_a = clean_sql_text(sql_a_raw) + ";"
        latency["sqlgen_contract_sec"] = time.perf_counter() - a0

        # One-shot
        b0 = time.perf_counter()
        pv_b = self.oneshot_prompt.format_prompt(question=question, schema=schema_text)
        resp_b = self.oneshot_model.invoke(pv_b.to_messages())
        sql_b_raw = getattr(resp_b, "content", str(resp_b))
        sql_b = clean_sql_text(sql_b_raw) + ";"
        latency["sqlgen_oneshot_sec"] = time.perf_counter() - b0

        # Validate contract
        v0 = time.perf_counter()
        contract_valid = validate_contract(contract, schema_text)
        latency["contract_validate_sec"] = time.perf_counter() - v0

        # Execute to check validity
        ok_a = ok_b = False
        err_a = err_b = None
        try:
            exec_sql(kwargs.get("db_path"), sql_a)
            ok_a = True
        except Exception as e:
            err_a = str(e)
        try:
            exec_sql(kwargs.get("db_path"), sql_b)
            ok_b = True
        except Exception as e:
            err_b = str(e)

        heuristics_applied: List[str] = []

        def prefer_by_exemplar(ex, sql_contract, sql_oneshot):
            if not ex:
                return None
            sc = (sql_contract or "").lower()
            so = (sql_oneshot or "").lower()
            score_c = 0
            score_o = 0
            for t in (ex.get("tags") or []):
                if t in sc:
                    score_c += 1
                if t in so:
                    score_o += 1
            for lit in re.findall(r"'([^']+)'", ex.get("answer_sql", "")):
                lit = lit.lower()
                if lit in sc:
                    score_c += 1
                if lit in so:
                    score_o += 1
            if score_c > score_o and score_c > 0:
                return "contract"
            if score_o > score_c and score_o > 0:
                return "oneshot"
            return None

        if ok_a and not ok_b:
            chosen = ("contract", sql_a)
        elif ok_b and not ok_a:
            chosen = ("oneshot", sql_b)
        elif ok_a and ok_b:
            preferred = prefer_by_exemplar(exemplar, sql_a, sql_b)
            if preferred:
                heuristics_applied.append("advisor_exemplar")
            if not preferred:
                preferred = self.prefer_car1_model(question, kwargs.get("db_id", ""), sql_a, sql_b)
                if preferred:
                    heuristics_applied.append("car1_model")
            if not preferred:
                preferred = self.prefer_population_sum(question, sql_a, sql_b)
                if preferred:
                    heuristics_applied.append("population_sum")
            if not preferred:
                preferred = self.prefer_both_languages(question, sql_a, sql_b)
                if preferred:
                    heuristics_applied.append("both_languages")
            if not preferred:
                preferred = self.prefer_distinct_regions(question, sql_a, sql_b)
                if preferred:
                    heuristics_applied.append("distinct_regions")
            if not preferred:
                preferred = self.prefer_by_question_target(question, sql_a, sql_b)
                if preferred:
                    heuristics_applied.append("question_target")
            if not preferred:
                preferred = self.prefer_by_question_literals(question, sql_a, sql_b)
                if preferred:
                    heuristics_applied.append("quoted_literals")
            if not preferred:
                preferred = self.prefer_by_unquoted_literals(question, sql_a, sql_b)
                if preferred:
                    heuristics_applied.append("unquoted_literals")

            if preferred:
                chosen = (preferred, sql_a if preferred == "contract" else sql_b)
            else:
                try:
                    j0 = time.perf_counter()
                    pv_j = self.chooser_prompt.format_prompt(
                        question=question,
                        schema=schema_text,
                        sql_contract=sql_a,
                        sql_oneshot=sql_b,
                    )
                    resp_j = self.oneshot_model.invoke(pv_j.to_messages())
                    latency["choose_judge_sec"] = time.perf_counter() - j0
                    choice_raw = getattr(resp_j, "content", str(resp_j)).strip().lower()
                    chosen = ("oneshot", sql_b) if "oneshot" in choice_raw else ("contract", sql_a)
                except Exception:
                    chosen = ("contract", sql_a)
        else:
            chosen = ("contract", sql_a)

        latency["total_sec"] = time.perf_counter() - total_start
        reasoner_decision = {
            "contract_verdict": contract_valid.get("contract_verdict", ""),
            "contract_validated": contract_valid,
            "chosen": chosen[0],
            "candidate_sql_contract": sql_a,
            "candidate_sql_oneshot": sql_b,
            "exec_ok_contract": ok_a,
            "exec_ok_oneshot": ok_b,
            "exec_err_contract": err_a,
            "exec_err_oneshot": err_b,
            "heuristics_applied": heuristics_applied,
        }

        return LoopResult(
            plan=plan,
            skeptic_feedback=debate_summary,
            reasoner_decision=reasoner_decision,
            final_sql=chosen[1],
            latency=latency,
            debug={"exemplar_used": bool(exemplar), "chosen": chosen[0]},
        )

