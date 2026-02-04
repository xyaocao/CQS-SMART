"""
Agents:
1. PlannerAgentV2 - Plan with join reasoning
2. SQLGenAgent - Generate SQL from plan
3. Skeptic - Combined review (Skeptic + Entity + Confidence)
4. SQLGenWithFeedbackAgent - Regenerate SQL with feedback
5. SQLRefinerAgent - Fix execution errors only
6. SelfConsistencyVoter - Vote on multiple candidates
"""
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import json
import re
import sys
import time
from pathlib import Path
from collections import Counter
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))

from baseline.llm import LLMConfig, get_llm_chat_model
from baseline.parse import json_file, get_response_text, extract_sql, log_raw_response

from prompts_secondloop import (
    CRITICAL_QUESTIONS_SECONDLOOP,
    Planner_system_prompt_v2, Planner_human_v2,
    SQLGen_system_prompt_v2, SQLGen_human_v2,
    SQLGen_with_feedback_system, SQLGen_with_feedback_human,
    SQLReviewer_system_prompt, SQLReviewer_human,
    SQLRefiner_system_prompt, SQLRefiner_human,
    SelfConsistencyVoter_system_prompt, SelfConsistencyVoter_human,
    SelfVerification_system_prompt, SelfVerification_human,
)


# =============================================================================
# STATE CLASSES
# =============================================================================

@dataclass
class PlannerState:
    """State for Planner."""
    question: str
    schema_text: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    latency: float = 0.0


@dataclass
class SQLGenState:
    """State for SQL Generator."""
    question: str
    schema_text: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    sql: str = ""
    raw: str = ""
    latency: float = 0.0


@dataclass
class SQLReviewerState:
    """State for SQL Reviewer (combined Skeptic + Entity + Confidence)."""
    question: str
    schema_text: str = ""
    sql: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    latency: float = 0.0


@dataclass
class SQLGenWithFeedbackState:
    """State for SQL Generator with feedback."""
    question: str
    schema_text: str = ""
    previous_sql: str = ""
    issues: List[str] = field(default_factory=list)
    revision_hints: List[str] = field(default_factory=list)
    sql: str = ""
    raw: str = ""
    latency: float = 0.0


@dataclass
class SQLRefinerState:
    """State for SQL Refiner (execution errors only)."""
    question: str
    schema_text: str = ""
    failed_sql: str = ""
    error: str = ""
    fixed_sql: str = ""
    analysis: str = ""
    raw: str = ""
    latency: float = 0.0


# =============================================================================
# PLANNER AGENT
# =============================================================================

class PlannerAgentV2:
    """Planner with explicit join logic rules."""

    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", Planner_system_prompt_v2),
            ("human", Planner_human_v2),
        ])

        graph = StateGraph(PlannerState)
        graph.add_node("plan", self._plan_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", END)
        self.app = graph.compile()

    def _plan_node(self, state: PlannerState) -> PlannerState:
        start = time.perf_counter()

        prompt_value = self.prompt.format_prompt(
            question=state.question,
            schema=state.schema_text,
        )

        response = self.model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start

        content = get_response_text(response)
        state.plan = json_file(content)
        state.raw = content

        if isinstance(state.plan, dict):
            state.plan.setdefault("tables", [])
            state.plan.setdefault("columns", [])
            state.plan.setdefault("joins", [])
            state.plan.setdefault("join_reasoning", "")

        return state

    def run(self, question: str, schema_text: str) -> Tuple[Dict[str, Any], float, str]:
        """Returns (plan, latency, raw_output)."""
        state = PlannerState(question=question, schema_text=schema_text or "")
        out = PlannerState(**self.app.invoke(state))
        return out.plan, out.latency, out.raw


# =============================================================================
# SQL GENERATOR AGENT
# =============================================================================

class SQLGenAgent:
    """SQL Generator from plan."""

    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SQLGen_system_prompt_v2),
            ("human", SQLGen_human_v2),
        ])

        graph = StateGraph(SQLGenState)
        graph.add_node("generate", self._generate_node)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", END)
        self.app = graph.compile()

    def _generate_node(self, state: SQLGenState) -> SQLGenState:
        start = time.perf_counter()

        prompt_value = self.prompt.format_prompt(
            question=state.question,
            schema=state.schema_text,
            plan=json.dumps(state.plan, indent=2, ensure_ascii=False),
        )

        response = self.model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start

        content = get_response_text(response)
        state.sql = extract_sql(content)
        state.raw = content

        return state

    def run(self, question: str, schema_text: str, plan: Dict[str, Any]) -> Tuple[str, float, str]:
        """Returns (sql, latency, raw_output)."""
        state = SQLGenState(
            question=question,
            schema_text=schema_text or "",
            plan=plan or {}
        )
        out = SQLGenState(**self.app.invoke(state))
        return out.sql, out.latency, out.raw


# =============================================================================
# SQL REVIEWER AGENT (Combined: Skeptic + Entity + Confidence)
# =============================================================================

class Skeptic:
    """
    Combined SQL Reviewer - replaces separate Skeptic, EntityValidator, ConfidenceScorer.

    Reviews SQL for:
    1. Entity-attribute alignment
    2. Table/join correctness
    3. Aggregation semantics
    4. Condition correctness
    5. SQL structure

    Outputs verdict, confidence, issues, and revision hints.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.critical_questions = CRITICAL_QUESTIONS_SECONDLOOP

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SQLReviewer_system_prompt),
            ("human", SQLReviewer_human),
        ])

        graph = StateGraph(SQLReviewerState)
        graph.add_node("review", self._review_node)
        graph.add_edge(START, "review")
        graph.add_edge("review", END)
        self.app = graph.compile()

    def _review_node(self, state: SQLReviewerState) -> SQLReviewerState:
        start = time.perf_counter()

        prompt_value = self.prompt.format_prompt(
            question=state.question,
            schema=state.schema_text,
            plan=json.dumps(state.plan, indent=2, ensure_ascii=False),
            sql=state.sql,
            critical_questions=self.critical_questions,
        )

        response = self.model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start

        content = get_response_text(response)
        review = json_file(content)
        state.raw = content

        # Ensure required fields
        # CRITICAL: If parsing fails or review is malformed, default to "needs_revision"
        # to be conservative - better to regenerate than pass through bad SQL
        if isinstance(review, dict):
            # If we got a dict but missing verdict, be conservative
            if "verdict" not in review:
                review["verdict"] = "needs_revision"
                review["confidence"] = 0.3
                review.setdefault("issues", ["Review output incomplete - regenerate for safety"])
                review.setdefault("revision_hints", ["Re-verify entity-attribute alignment"])
            else:
                review.setdefault("confidence", 0.5)
                review.setdefault("issues", [])
                review.setdefault("revision_hints", [])
            review.setdefault("analysis", {})
        else:
            # Parsing completely failed - be very conservative
            review = {
                "verdict": "needs_revision",
                "confidence": 0.3,
                "issues": ["Review parsing failed - SQL needs verification"],
                "revision_hints": ["Regenerate SQL with careful entity-attribute alignment"],
                "analysis": {"parse_error": True}
            }

        state.review = review
        return state

    def run(self, question: str, schema_text: str, sql: str, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], float, str]:
        """
        Returns (review, latency, raw_output).

        review = {
            "verdict": "ok" | "needs_revision",
            "confidence": 0.0-1.0,
            "issues": [...],
            "revision_hints": [...],
            "analysis": {...}
        }
        """
        state = SQLReviewerState(
            question=question,
            schema_text=schema_text or "",
            sql=sql,
            plan=plan or {}
        )
        out = SQLReviewerState(**self.app.invoke(state))
        return out.review, out.latency, out.raw


# =============================================================================
# SQL GENERATOR WITH FEEDBACK (for regeneration)
# =============================================================================

class SQLGenWithFeedbackAgent:
    """
    SQL Generator that takes feedback from reviewer and regenerates SQL.

    This is used instead of surgical correction - cleaner and more reliable.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SQLGen_with_feedback_system),
            ("human", SQLGen_with_feedback_human),
        ])

        graph = StateGraph(SQLGenWithFeedbackState)
        graph.add_node("regenerate", self._regenerate_node)
        graph.add_edge(START, "regenerate")
        graph.add_edge("regenerate", END)
        self.app = graph.compile()

    def _regenerate_node(self, state: SQLGenWithFeedbackState) -> SQLGenWithFeedbackState:
        start = time.perf_counter()

        prompt_value = self.prompt.format_prompt(
            question=state.question,
            schema=state.schema_text,
            previous_sql=state.previous_sql,
            issues="\n".join(f"- {issue}" for issue in state.issues),
            revision_hints="\n".join(f"- {hint}" for hint in state.revision_hints),
        )

        response = self.model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start

        content = get_response_text(response)
        state.sql = extract_sql(content)
        state.raw = content

        return state

    def run(self, question: str, schema_text: str, previous_sql: str,
            issues: List[str], revision_hints: List[str]) -> Tuple[str, float, str]:
        """Returns (sql, latency, raw_output)."""
        state = SQLGenWithFeedbackState(
            question=question,
            schema_text=schema_text or "",
            previous_sql=previous_sql,
            issues=issues or [],
            revision_hints=revision_hints or [],
        )
        out = SQLGenWithFeedbackState(**self.app.invoke(state))
        return out.sql, out.latency, out.raw


# =============================================================================
# SQL REFINER AGENT (for execution errors only)
# =============================================================================

class SQLRefinerAgent:
    """
    SQL Refiner - fixes SQL that failed during execution.

    Only used when actual execution fails, not for semantic issues.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SQLRefiner_system_prompt),
            ("human", SQLRefiner_human),
        ])

        graph = StateGraph(SQLRefinerState)
        graph.add_node("refine", self._refine_node)
        graph.add_edge(START, "refine")
        graph.add_edge("refine", END)
        self.app = graph.compile()

    def _refine_node(self, state: SQLRefinerState) -> SQLRefinerState:
        start = time.perf_counter()

        prompt_value = self.prompt.format_prompt(
            question=state.question,
            schema=state.schema_text,
            failed_sql=state.failed_sql,
            error=state.error,
        )

        response = self.model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start

        content = get_response_text(response)
        result = json_file(content)
        state.raw = content

        if isinstance(result, dict):
            state.fixed_sql = result.get("fixed_sql", state.failed_sql)
            state.analysis = result.get("analysis", "")
        else:
            state.fixed_sql = extract_sql(content)
            state.analysis = "Could not parse refiner output"

        return state

    def run(self, question: str, schema_text: str, failed_sql: str, error: str) -> Tuple[str, str, float, str]:
        """Returns (fixed_sql, analysis, latency, raw_output)."""
        state = SQLRefinerState(
            question=question,
            schema_text=schema_text or "",
            failed_sql=failed_sql,
            error=error,
        )
        out = SQLRefinerState(**self.app.invoke(state))
        return out.fixed_sql, out.analysis, out.latency, out.raw


# =============================================================================
# SELF-CONSISTENCY VOTER
# =============================================================================

class SelfConsistencyVoter:
    """Votes on multiple SQL candidates."""

    def __init__(self, llm_config: LLMConfig = None):
        self.llm_config = llm_config
        self.model = get_llm_chat_model(llm_config) if llm_config else None
        if self.model:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", SelfConsistencyVoter_system_prompt),
                ("human", SelfConsistencyVoter_human),
            ])

    def simple_vote(self, sql_candidates: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Simple majority voting without LLM.
        Returns (winning_sql, vote_info).
        """
        if not sql_candidates:
            return "", {"error": "no candidates"}

        def normalize(sql: str) -> str:
            return ' '.join(sql.lower().split())

        normalized = [normalize(sql) for sql in sql_candidates]
        counts = Counter(normalized)
        most_common_normalized, vote_count = counts.most_common(1)[0]

        for sql, norm in zip(sql_candidates, normalized):
            if norm == most_common_normalized:
                return sql, {
                    "method": "simple_majority",
                    "vote_count": vote_count,
                    "total_candidates": len(sql_candidates),
                    "unique_candidates": len(counts),
                }

        return sql_candidates[0], {"method": "fallback"}

    def vote_with_llm(self, question: str, schema_text: str, sql_candidates: List[str]) -> Tuple[str, Dict[str, Any], float]:
        """
        LLM-based voting.
        Returns (winning_sql, vote_info, latency).
        """
        if not self.model:
            winner, info = self.simple_vote(sql_candidates)
            return winner, info, 0.0

        start = time.perf_counter()

        candidates_text = "\n\n".join([
            f"### Candidate {i}:\n{sql}"
            for i, sql in enumerate(sql_candidates)
        ])

        prompt_value = self.prompt.format_prompt(
            question=question,
            schema=schema_text,
            candidates=candidates_text,
        )

        response = self.model.invoke(prompt_value.to_messages())
        latency = time.perf_counter() - start

        content = get_response_text(response)
        result = json_file(content)

        if isinstance(result, dict):
            winning_sql = result.get("winning_sql", sql_candidates[0] if sql_candidates else "")
            winning_index = result.get("winning_index", 0)
            reasoning = result.get("reasoning", "")
        else:
            winning_sql = sql_candidates[0] if sql_candidates else ""
            winning_index = 0
            reasoning = ""

        return winning_sql, {
            "method": "llm_vote",
            "winning_index": winning_index,
            "reasoning": reasoning,
            "candidates": sql_candidates,
        }, latency


# =============================================================================
# SELF-VERIFICATION AGENT (Solution 4)
# =============================================================================

@dataclass
class SelfVerificationState:
    """State for Self-Verification Agent."""
    question: str
    schema_text: str = ""
    sql: str = ""
    verification: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    latency: float = 0.0


class SelfVerificationAgent:
    """
    Self-Verification Agent - verifies if SQL output matches question intent.
    
    This is an additional safety check after the reviewer passes a query.
    It predicts what the SQL will return and checks if it matches what
    the question is asking for.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SelfVerification_system_prompt),
            ("human", SelfVerification_human),
        ])

        graph = StateGraph(SelfVerificationState)
        graph.add_node("verify", self._verify_node)
        graph.add_edge(START, "verify")
        graph.add_edge("verify", END)
        self.app = graph.compile()

    def _verify_node(self, state: SelfVerificationState) -> SelfVerificationState:
        start = time.perf_counter()

        prompt_value = self.prompt.format_prompt(
            question=state.question,
            sql=state.sql,
            schema=state.schema_text,
        )

        response = self.model.invoke(prompt_value.to_messages())
        state.latency = time.perf_counter() - start

        content = get_response_text(response)
        verification = json_file(content)
        state.raw = content

        # Ensure required fields with conservative defaults
        if isinstance(verification, dict):
            verification.setdefault("sql_will_return", "")
            verification.setdefault("question_asks_for", "")
            verification.setdefault("match", False)  # Conservative: assume mismatch if not specified
            verification.setdefault("mismatch_reason", "Verification incomplete")
        else:
            # Parsing failed - be conservative
            verification = {
                "sql_will_return": "",
                "question_asks_for": "",
                "match": False,
                "mismatch_reason": "Verification parsing failed - assume mismatch for safety",
                "parse_error": True
            }

        state.verification = verification
        return state

    def run(self, question: str, schema_text: str, sql: str) -> Tuple[Dict[str, Any], float, str]:
        """
        Returns (verification, latency, raw_output).

        verification = {
            "sql_will_return": "description of SQL output",
            "question_asks_for": "description of question intent",
            "match": True/False,
            "mismatch_reason": "reason if match is False"
        }
        """
        state = SelfVerificationState(
            question=question,
            schema_text=schema_text or "",
            sql=sql,
        )
        out = SelfVerificationState(**self.app.invoke(state))
        return out.verification, out.latency, out.raw
