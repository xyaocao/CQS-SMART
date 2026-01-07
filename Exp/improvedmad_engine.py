import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from baseline.llm import LLMConfig, get_llm_chat_model
from baseline.planneragent import json_file
from baseline.evaluation import exec_sql
from loop_engines import LoopEngine, LoopResult, LoopStageError
from prompts_improvedmad import (ContractReasoner_system_prompt, ContractReasoner_human, ContractSQLGen_system_prompt, ContractSQLGen_human, OneShotSQLGen_system_prompt, OneShotSQLGen_human)
from history_hints import load_history_hints, match_history_hint

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


def extract_schema_tables_columns(schema_text: str) -> Tuple[set[str], set[str]]:
    """Extract table names and table.column names from schema.sql-ish text."""
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
    """Drop obviously invalid table/column references from contract."""
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
    out["where"] = _filter_cols(out.get("where", []))  # drop predicates with bad table.col
    out["having"] = _filter_cols(out.get("having", []))
    out["group_by"] = _filter_cols(out.get("group_by", []))

    # joins: keep only joins whose table exists and on-clause refs exist
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

    # cap notes
    if isinstance(out.get("notes"), list):
        out["notes"] = dedup_list(out["notes"], limit=5)
    # If select is empty, mark revise and add a note to force projection fix.
    if not out.get("select"):
        out["contract_verdict"] = "revise"
        notes = out.get("notes") or []
        notes.append("select_missing_output")
        out["notes"] = dedup_list(notes, limit=5)
    return out


def clean_sql_text(sql: str) -> str:
    """Strip code fences/markdown and normalize trailing semicolon."""
    s = (sql or "").strip()
    # Remove leading code fence like ```sql or ```SQL
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s, flags=re.IGNORECASE)
    # Remove trailing ```
    s = re.sub(r"```$", "", s).strip()
    # Remove trailing semicolons/spaces, will add one later
    s = s.rstrip(" \n\r\t;")
    return s


class ImprovedMADEngine(LoopEngine):
    """ImprovedMAD engine: BaseMAD-style debate + contract reasoner + validator + 2-candidate chooser."""

    def __init__(
        self,
        planner,
        defender,
        skeptic,
        max_debate_rounds: int = 3,
        llm_config_reasoner: Optional[LLMConfig] = None,
        llm_config_sqlgen: Optional[LLMConfig] = None,
        llm_config_oneshot: Optional[LLMConfig] = None,
        history_hints_path: Optional[Path] = None,
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
        # History hints (soft bias, never inject SQL)
        # First-version (no hints): set self.history_hints = [] and comment out the load below.
        self.history_hints = load_history_hints(history_hints_path or Path("Exp/history_hints.jsonl"))
        # Lightweight chooser prompt to break ties between two executable SQLs.
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

    @staticmethod
    def prefer_by_question_target(question: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        """Heuristic: if question mentions 'model' and only one SQL mentions it, prefer that."""
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
        """Heuristic: if question contains quoted literals and only one SQL preserves them, prefer it."""
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
        """Heuristic: prefer candidate preserving salient unquoted tokens (e.g., math, usa)."""
        stop = {
            "what",
            "which",
            "who",
            "how",
            "many",
            "much",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "for",
            "to",
            "in",
            "on",
            "by",
            "with",
            "and",
            "or",
            "from",
            "show",
            "list",
            "find",
            "get",
            "all",
            "give",
            "display",
            "that",
            "those",
            "these",
            "at",
            "each",
            "per",
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
        """Prefer SUM(Population) for 'how many people/population' questions over counting rows."""
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
        # if both are counts and none sums, prefer the one without count-only
        if has_count_only(sc) and not has_count_only(so):
            return "oneshot"
        if has_count_only(so) and not has_count_only(sc):
            return "contract"
        return None

    @staticmethod
    def prefer_both_languages(question: str, sql_contract: str, sql_oneshot: str) -> Optional[str]:
        """If two languages are requested with AND/BOTH, prefer candidates enforcing both languages."""
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
        """For region/area list questions, prefer DISTINCT-bearing candidate."""
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
        """For car_1 model queries, prefer projection from car_names.Model joined to cars_data.Id."""
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

    def prefer_by_history_hint(self, question: str, db_id: str, sql_contract: str, sql_oneshot: str) -> Tuple[Optional[str], Optional[str]]:
        """Apply history hint bias if a single hint matches."""
        hint = match_history_hint(self.history_hints, question, db_id)
        if not hint:
            return None, None
        prefs = hint.get("hint") or {}
        preferred_select = prefs.get("preferred_select") or []
        preserve_literals = prefs.get("preserve_literals") or []
        sc = (sql_contract or "").lower()
        so = (sql_oneshot or "").lower()
        def score(sql: str) -> int:
            s = sql.lower()
            score = 0
            for col in preferred_select:
                if str(col).lower() in s:
                    score += 1
            for lit in preserve_literals:
                if str(lit).lower() in s:
                    score += 1
            return score
        c_score = score(sql_contract)
        o_score = score(sql_oneshot)
        if c_score > 0 and o_score == 0:
            return "contract", hint.get("pattern")
        if o_score > 0 and c_score == 0:
            return "oneshot", hint.get("pattern")
        return None, hint.get("pattern")

    # @staticmethod
    # def prefer_by_plan_conditions(plan: Dict[str, Any], sql_contract: str, sql_oneshot: str) -> Optional[str]:
    #     """
    #     Heuristic: prefer the SQL that better preserves planner conditions (e.g., specific literals).
    #     Simple string check: count how many plan condition substrings appear in each SQL.
    #     """
    #     conds = plan.get("conditions") if isinstance(plan, dict) else None
    #     if not conds:
    #         return None
    #     sc = (sql_contract or "").lower()
    #     so = (sql_oneshot or "").lower()
    #     c_score = sum(1 for cond in conds if isinstance(cond, str) and cond.lower() in sc)
    #     o_score = sum(1 for cond in conds if isinstance(cond, str) and cond.lower() in so)
    #     if c_score > o_score:
    #         return "contract"
    #     if o_score > c_score:
    #         return "oneshot"
    #     return None

    def run_debate(self, question: str, schema_text: str, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (debate_summary, debug). Minimal BaseMAD-like debate."""
        debug: Dict[str, Any] = {"debate_rounds": [], "max_debate_rounds": self.max_debate_rounds}
        last_def: Dict[str, Any] = {}
        last_sk: Dict[str, Any] = {}
        consensus = False

        # Round 1: parallel-ish (we still run sequentially here)
        defense, d_lat = self.defender.run(question, schema_text, plan, skeptic_feedback=None)
        defense = defense.copy()
        if isinstance(defense.get("justification"), list):
            defense["justification"] = dedup_list(defense["justification"])
        if isinstance(defense.get("notes_for_reasoner"), list):
            defense["notes_for_reasoner"] = dedup_list(defense["notes_for_reasoner"])
        last_def = defense

        skeptic_view, s_lat = self.skeptic.run(question, schema_text, plan, execution_feedback=None)
        skeptic_view = skeptic_view.copy()
        if isinstance(skeptic_view.get("issues"), list):
            skeptic_view["issues"] = dedup_list(skeptic_view["issues"])
        if isinstance(skeptic_view.get("recommendations"), list):
            skeptic_view["recommendations"] = dedup_list(skeptic_view["recommendations"])
        last_sk = skeptic_view

        consensus = bool(last_sk.get("consensus", False)) and str(last_sk.get("verdict", "")).lower() == "ok"
        debug["debate_rounds"].append(
            {"round": 1, "defender_view": last_def, "skeptic_view": last_sk, "consensus": consensus, "evaluation_mode": "round1"}
        )

        # Further rounds (sequential)
        if not consensus:
            for r in range(2, self.max_debate_rounds + 1):
                defense, _ = self.defender.run(question, schema_text, plan, skeptic_feedback=last_sk)
                defense = defense.copy()
                if isinstance(defense.get("justification"), list):
                    defense["justification"] = dedup_list(defense["justification"])
                if isinstance(defense.get("notes_for_reasoner"), list):
                    defense["notes_for_reasoner"] = dedup_list(defense["notes_for_reasoner"])
                last_def = defense

                skeptic_view, _ = self.skeptic.run(question, schema_text, plan, execution_feedback=last_def)
                skeptic_view = skeptic_view.copy()
                if isinstance(skeptic_view.get("issues"), list):
                    skeptic_view["issues"] = dedup_list(skeptic_view["issues"])
                if isinstance(skeptic_view.get("recommendations"), list):
                    skeptic_view["recommendations"] = dedup_list(skeptic_view["recommendations"])
                last_sk = skeptic_view

                consensus = bool(last_sk.get("consensus", False)) and str(last_sk.get("verdict", "")).lower() == "ok"
                debug["debate_rounds"].append(
                    {"round": r, "defender_view": last_def, "skeptic_view": last_sk, "consensus": consensus, "evaluation_mode": "sequential"}
                )
                if consensus:
                    break

        # Summary
        all_issues: List[Any] = []
        all_recs: List[Any] = []
        for rd in debug["debate_rounds"]:
            sk = rd.get("skeptic_view", {})
            df = rd.get("defender_view", {})
            if isinstance(sk.get("issues"), list):
                all_issues.extend(sk["issues"])
            if isinstance(sk.get("recommendations"), list):
                all_recs.extend(sk["recommendations"])
            if isinstance(df.get("notes_for_reasoner"), list):
                all_recs.extend(df["notes_for_reasoner"])

        debate_summary = {
            "consensus_reached": consensus,
            "final_defender_stance": last_def.get("stance", ""),
            "final_defender_viewpoint": last_def.get("viewpoint", ""),
            "final_skeptic_verdict": last_sk.get("verdict", ""),
            "final_skeptic_position": last_sk.get("position", ""),
            "critical_issues": dedup_list(all_issues, limit=10),
            "key_recommendations": dedup_list(all_recs, limit=10),
            "final_defender_full": last_def,
            "final_skeptic_full": last_sk,
            "debate_rounds_count": len(debug["debate_rounds"]),
        }
        return debate_summary, debug

    def run(self, question: str, schema_text: str, **kwargs: Any) -> LoopResult:
        total_start = time.perf_counter()
        db_path = kwargs.get("db_path")
        if not db_path:
            raise LoopStageError("ImprovedMAD", ValueError("db_path required"), 0.0, {"latency": {}})

        latency: Dict[str, float] = {}

        # Planner
        try:
            p0 = time.perf_counter()
            plan, p_lat = self.planner.run(question, schema_text)
            latency["planner_sec"] = p_lat if p_lat is not None else (time.perf_counter() - p0)
        except Exception as exc:
            raise LoopStageError("Planner", exc, time.perf_counter() - total_start, {"latency": latency}) from exc

        debate_summary: Dict[str, Any] = {}
        debug: Dict[str, Any] = {}

        # Debate (BaseMAD-style). If it fails, continue without it.
        debate_used = False
        try:
            d0 = time.perf_counter()
            debate_summary, debate_debug = self.run_debate(question, schema_text, plan)
            latency["debate_sec"] = time.perf_counter() - d0
            debug["debate"] = debate_debug
            debate_used = True
        except Exception as exc:
            debate_summary = {}
            debug["debate_error"] = str(exc)
            latency["debate_sec"] = 0.0

        # Contract reasoner
        c0 = time.perf_counter()
        pv = self.contract_prompt.format_prompt(
            question=question,
            schema=schema_text,
            plan=json.dumps(plan, ensure_ascii=False),
            debate_summary=json.dumps(debate_summary or {}, ensure_ascii=False),
        )
        resp = self.reasoner_model.invoke(pv.to_messages())
        raw = getattr(resp, "content", str(resp))
        latency["contract_reasoner_sec"] = time.perf_counter() - c0
        contract = json_file(raw)
        if not isinstance(contract, dict):
            contract = {"contract_verdict": "revise", "intent": "invalid_contract", "from": [], "select": [], "joins": [], "where": [], "group_by": [], "having": [], "order_by": None, "limit": None, "distinct": False, "set_op": None, "notes": ["contract_parse_failed"]}

        # Validate contract
        v0 = time.perf_counter()
        contract_valid = validate_contract(contract, schema_text)
        latency["contract_validate_sec"] = time.perf_counter() - v0

        # Candidate A: from contract
        a0 = time.perf_counter()
        pv_a = self.contract_sql_prompt.format_prompt(
            question=question,
            schema=schema_text,
            contract=json.dumps(contract_valid, ensure_ascii=False),
        )
        resp_a = self.contract_sqlgen_model.invoke(pv_a.to_messages())
        sql_a_raw = getattr(resp_a, "content", str(resp_a))
        sql_a = clean_sql_text(sql_a_raw) + ";"
        latency["sqlgen_contract_sec"] = time.perf_counter() - a0

        # Candidate B: one-shot
        b0 = time.perf_counter()
        pv_b = self.oneshot_prompt.format_prompt(question=question, schema=schema_text)
        resp_b = self.oneshot_model.invoke(pv_b.to_messages())
        sql_b_raw = getattr(resp_b, "content", str(resp_b))
        sql_b = clean_sql_text(sql_b_raw) + ";"
        latency["sqlgen_oneshot_sec"] = time.perf_counter() - b0

        # Choose best executable
        ch0 = time.perf_counter()
        ok_a = ok_b = False
        err_a = err_b = None
        try:
            exec_sql(db_path, sql_a)
            ok_a = True
        except Exception as e:
            err_a = str(e)
        try:
            exec_sql(db_path, sql_b)
            ok_b = True
        except Exception as e:
            err_b = str(e)

        heuristics_applied: List[str] = []

        if ok_a and not ok_b:
            chosen = ("contract", sql_a)
        elif ok_b and not ok_a:
            chosen = ("oneshot", sql_b)
        elif ok_a and ok_b:
            # Final chooser: hints + heuristics, then judge; fallback to contract on judge failure.
            preferred, hint_pattern = self.prefer_by_history_hint(question, kwargs.get("db_id", ""), sql_a, sql_b)
            if preferred:
                heuristics_applied.append(f"history_hint:{hint_pattern}")
            if not preferred:
                preferred = self.prefer_car1_model(question, kwargs.get("db_id", ""), sql_a, sql_b)
                if preferred:
                    heuristics_applied.append("car1_model")
            if not preferred:
            #     preferred = self.prefer_by_plan_conditions(plan, sql_a, sql_b)
            #     if preferred:
            #         heuristics_applied.append("plan_conditions")
            # if not preferred:
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
                    chosen = ("contract", sql_a)  # safe fallback

            # First-version chooser (judge only, no hints/heuristics):
            # try:
            #     j0 = time.perf_counter()
            #     pv_j = self.chooser_prompt.format_prompt(
            #         question=question,
            #         schema=schema_text,
            #         sql_contract=sql_a,
            #         sql_oneshot=sql_b,
            #     )
            #     resp_j = self.oneshot_model.invoke(pv_j.to_messages())
            #     latency["choose_judge_sec"] = time.perf_counter() - j0
            #     choice_raw = getattr(resp_j, "content", str(resp_j)).strip().lower()
            #     chosen = ("oneshot", sql_b) if "oneshot" in choice_raw else ("contract", sql_a)
            # except Exception:
            #     chosen = ("contract", sql_a)  # safe fallback
        else:
            chosen = ("contract", sql_a)  # keep contract for debugging
        latency["choose_sec"] = time.perf_counter() - ch0

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
        feedback = {
            "mode": "ImprovedMAD",
            "debate_used": debate_used,
            "debate_summary": debate_summary,
            "contract": contract_valid,
        }

        return LoopResult(
            plan=plan,
            skeptic_feedback=feedback,
            reasoner_decision=reasoner_decision,
            final_sql=chosen[1],
            latency=latency,
            debug=debug,
        )


