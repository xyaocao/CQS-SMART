"""
Simplified SecondLoop Engine.

Pipeline:
1. Planner → Plan (with join reasoning)
2. SQLGen → initial_sql (with optional voting)
3. SQL Reviewer → verdict, confidence, issues, hints
4. If needs_revision → SQLGen with feedback → revised_sql
5. (Optional) Execute → if error → Refiner → final_sql

Key benefits:
- Single combined reviewer (vs separate Skeptic + Entity + Confidence)
- Feedback-based regeneration (vs surgical correction)
- 3-4 LLM calls (vs 7-8 in original design)
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SecondLoopResult:
    """Result from simplified SecondLoop engine."""
    plan: Dict[str, Any]
    initial_sql: str
    review: Dict[str, Any]  # Combined review (verdict, confidence, issues, hints)
    revised_sql: str  # SQL after regeneration (if any)
    final_sql: str
    latency: Dict[str, float]
    # Tracking
    revision_applied: bool = False
    refiner_applied: bool = False
    verification_applied: bool = False  # Whether self-verification was run
    verification: Dict[str, Any] | None = None  # Self-verification result
    voting_info: Dict[str, Any] | None = None
    debug: Dict[str, Any] | None = None


class SecondLoopEngine:
    """
    Simplified SecondLoop Engine.

    Pipeline:
    1. Planner → Plan
    2. SQLGen → initial_sql (with optional voting)
    3. SQL Reviewer → review
    4. If verdict="needs_revision" → SQLGenWithFeedback → revised_sql
    5. (Optional --refine) Execute → if error → Refiner

    LLM Calls:
    - No issues: Planner + SQLGen + Reviewer = 3 calls
    - With revision: + SQLGenWithFeedback = 4 calls
    - With voting (N=3): Planner + SQLGen*3 + Reviewer + maybe feedback = 5-6 calls
    """

    def __init__(
        self,
        planner,
        sql_gen,
        sql_reviewer,
        sql_gen_with_feedback,
        refiner=None,
        voter=None,
        verifier=None,  # SelfVerificationAgent (optional)
        enable_voting: bool = False,
        voting_samples: int = 3,
        enable_refiner: bool = False,
        enable_verification: bool = False,  # Whether to run self-verification
        db_executor=None,  # Function: (db_path, sql) -> (result, error)
    ):
        """
        Initialize simplified SecondLoop engine.

        Args:
            planner: PlannerAgentV2
            sql_gen: SQLGenAgent
            sql_reviewer: Skeptic (combined Skeptic + Entity + Confidence)
            sql_gen_with_feedback: SQLGenWithFeedbackAgent (for regeneration)
            refiner: SQLRefinerAgent (optional, for execution errors)
            voter: SelfConsistencyVoter (optional)
            verifier: SelfVerificationAgent (optional, for self-verification)
            enable_voting: Whether to generate multiple candidates and vote
            voting_samples: Number of SQL candidates for voting
            enable_refiner: Whether to run SQL and refine on execution error
            enable_verification: Whether to run self-verification after review
            db_executor: Function to execute SQL (for refiner)
        """
        self.planner = planner
        self.sql_gen = sql_gen
        self.sql_reviewer = sql_reviewer
        self.sql_gen_with_feedback = sql_gen_with_feedback
        self.refiner = refiner
        self.voter = voter
        self.verifier = verifier
        self.enable_voting = enable_voting
        self.voting_samples = voting_samples
        self.enable_refiner = enable_refiner
        self.enable_verification = enable_verification
        self.db_executor = db_executor

    def run(self, question: str, schema_text: str, db_path: str = None, **kwargs) -> SecondLoopResult:
        """Run the simplified SecondLoop pipeline."""
        total_start = time.perf_counter()
        latency = {}
        debug = {}
        voting_info = None

        # === STAGE 1: PLANNER ===
        try:
            plan, plan_lat, plan_raw = self.planner.run(question, schema_text)
            latency["planner_sec"] = plan_lat
            debug["planner_raw"] = plan_raw
            debug["join_reasoning"] = plan.get("join_reasoning", "")
        except Exception as e:
            plan = {"tables": [], "columns": [], "error": str(e)}
            latency["planner_sec"] = 0
            debug["planner_error"] = str(e)

        # === STAGE 2: SQL GENERATION (with optional voting) ===
        if self.enable_voting and self.voter:
            sql_candidates = []
            sql_latencies = []

            for i in range(self.voting_samples):
                try:
                    sql, sql_lat, _ = self.sql_gen.run(question, schema_text, plan)
                    sql_candidates.append(sql)
                    sql_latencies.append(sql_lat)
                except Exception as e:
                    debug[f"sqlgen_error_{i}"] = str(e)

            latency["sqlgen_sec"] = sum(sql_latencies)

            if sql_candidates:
                initial_sql, voting_info = self.voter.simple_vote(sql_candidates)
                voting_info["candidates"] = sql_candidates
            else:
                initial_sql = ""
                voting_info = {"error": "no candidates generated"}
        else:
            try:
                initial_sql, sql_lat, sql_raw = self.sql_gen.run(question, schema_text, plan)
                latency["sqlgen_sec"] = sql_lat
                debug["sqlgen_raw"] = sql_raw
            except Exception as e:
                initial_sql = ""
                latency["sqlgen_sec"] = 0
                debug["sqlgen_error"] = str(e)

        # === STAGE 3: SQL REVIEWER (Combined Skeptic + Entity + Confidence) ===
        try:
            review, review_lat, review_raw = self.sql_reviewer.run(
                question, schema_text, initial_sql, plan
            )
            latency["reviewer_sec"] = review_lat
            debug["reviewer_raw"] = review_raw
        except Exception as e:
            review = {
                "verdict": "ok",
                "confidence": 0.5,
                "issues": [],
                "revision_hints": [],
                "error": str(e)
            }
            latency["reviewer_sec"] = 0
            debug["reviewer_error"] = str(e)

        # === STAGE 4: REGENERATION IF NEEDED ===
        revised_sql = initial_sql
        revision_applied = False

        # Determine if regeneration is needed based on multiple criteria:
        # 1. Explicit "needs_revision" verdict
        # 2. Low confidence (< 0.85) even with "ok" verdict
        # 3. Any step check marked as "FAIL" in the analysis
        # 4. Parse error occurred
        verdict = review.get("verdict", "needs_revision")
        # Ensure confidence is a float (may come from LLM as string)
        confidence = float(review.get("confidence", 0.5))
        
        # Check for FAIL in step analysis (new CoT format)
        has_step_failure = False
        for key in ["step3_alignment_check", "step4_aggregation_check", 
                    "step5_exclusion_check", "step6_join_check"]:
            step_result = str(review.get(key, "")).upper()
            if "FAIL" in step_result:
                has_step_failure = True
                break
        
        # Also check legacy analysis format
        analysis = review.get("analysis", {})
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                if "FAIL" in str(value).upper():
                    has_step_failure = True
                    break
        
        # Trigger regeneration if any of these conditions are met
        needs_regeneration = (
            verdict == "needs_revision" or
            confidence < 0.85 or
            has_step_failure or
            review.get("parse_error", False) or
            analysis.get("parse_error", False)
        )

        if needs_regeneration:
            issues = review.get("issues", [])
            hints = review.get("revision_hints", [])
            
            # Add context about why regeneration was triggered
            if confidence < 0.85 and verdict == "ok":
                issues = issues + [f"Low confidence ({confidence:.2f}) - regenerating for safety"]
            if has_step_failure and verdict == "ok":
                issues = issues + ["Step analysis detected failure despite ok verdict"]
            
            # Add helpful hints based on detected issues
            issues_text = " ".join(str(i).lower() for i in issues)
            hints_text = " ".join(str(h).lower() for h in hints)
            
            # If alignment issue detected, add hint about checking column meanings
            if "alignment" in issues_text or "name" in issues_text or "column" in issues_text:
                if "column meaning" not in hints_text:
                    hints = hints + ["Check '### Column Meanings' in schema to find the correct column"]
            
            # If aggregation issue detected, add hint
            if "aggregation" in issues_text or "average" in issues_text or "avg" in issues_text:
                if "avg(" not in hints_text:
                    hints = hints + ["For computing averages, use AVG(column_name), not a pre-stored column"]
            
            # If entity mismatch detected, add hint
            if "entity" in issues_text or "song" in issues_text or "singer" in issues_text:
                if "meaning" not in hints_text:
                    hints = hints + ["Verify each column's meaning in schema matches the question's intent"]

            if issues or hints:
                try:
                    revised_sql, regen_lat, regen_raw = self.sql_gen_with_feedback.run(
                        question, schema_text, initial_sql, issues, hints
                    )
                    latency["regeneration_sec"] = regen_lat
                    debug["regeneration_raw"] = regen_raw
                    revision_applied = (revised_sql != initial_sql)
                except Exception as e:
                    latency["regeneration_sec"] = 0
                    debug["regeneration_error"] = str(e)

        final_sql = revised_sql
        refiner_applied = False
        verification_applied = False
        verification_result = None

        # === STAGE 4.5: SELF-VERIFICATION (Optional) ===
        # Run verification if enabled AND reviewer passed with "ok"
        # This is an additional safety check to catch mismatches
        if (self.enable_verification and self.verifier and 
            not needs_regeneration):  # Only verify if reviewer said "ok"
            try:
                verification_result, verify_lat, verify_raw = self.verifier.run(
                    question, schema_text, final_sql
                )
                latency["verification_sec"] = verify_lat
                debug["verification_raw"] = verify_raw
                verification_applied = True
                
                # If verification detects mismatch, trigger regeneration
                if not verification_result.get("match", True):
                    mismatch_reason = verification_result.get("mismatch_reason", "Verification detected mismatch")
                    debug["verification_triggered_regen"] = True
                    
                    try:
                        issues_for_regen = [f"Self-verification failed: {mismatch_reason}"]
                        hints_for_regen = [
                            f"SQL returns: {verification_result.get('sql_will_return', 'unknown')}",
                            f"Question asks for: {verification_result.get('question_asks_for', 'unknown')}",
                            "Fix the entity-attribute alignment"
                        ]
                        
                        verified_sql, verify_regen_lat, verify_regen_raw = self.sql_gen_with_feedback.run(
                            question, schema_text, final_sql, issues_for_regen, hints_for_regen
                        )
                        latency["verification_regen_sec"] = verify_regen_lat
                        debug["verification_regen_raw"] = verify_regen_raw
                        
                        if verified_sql and verified_sql != final_sql:
                            final_sql = verified_sql
                            revision_applied = True  # Mark as revised
                    except Exception as e:
                        debug["verification_regen_error"] = str(e)
            except Exception as e:
                debug["verification_error"] = str(e)

        # === STAGE 5: EXECUTION + REFINER (Optional) ===
        if self.enable_refiner and self.refiner and db_path and self.db_executor:
            try:
                exec_result, exec_error = self.db_executor(db_path, final_sql)

                if exec_error:
                    # SQL failed - try to fix it
                    try:
                        fixed_sql, analysis, refine_lat, refine_raw = self.refiner.run(
                            question, schema_text, final_sql, exec_error
                        )
                        latency["refiner_sec"] = refine_lat
                        debug["refiner_raw"] = refine_raw
                        debug["refiner_analysis"] = analysis
                        debug["execution_error"] = exec_error

                        if fixed_sql and fixed_sql != final_sql:
                            final_sql = fixed_sql
                            refiner_applied = True
                    except Exception as e:
                        debug["refiner_error"] = str(e)
            except Exception as e:
                debug["execution_exception"] = str(e)

        latency["total_sec"] = time.perf_counter() - total_start

        return SecondLoopResult(
            plan=plan,
            initial_sql=initial_sql,
            review=review,
            revised_sql=revised_sql,
            final_sql=final_sql,
            latency=latency,
            revision_applied=revision_applied,
            refiner_applied=refiner_applied,
            verification_applied=verification_applied,
            verification=verification_result,
            voting_info=voting_info,
            debug=debug,
        )


class DirectSQLEngine:
    """
    Direct SQL generation (bypass planner and reviewer).
    For simple queries or baseline comparison.
    """

    def __init__(self, sql_gen):
        self.sql_gen = sql_gen

    def run(self, question: str, schema_text: str, **kwargs) -> SecondLoopResult:
        """Direct SQL generation."""
        total_start = time.perf_counter()

        try:
            sql, sql_lat, sql_raw = self.sql_gen.run(question, schema_text, {})
        except Exception as e:
            sql = ""
            sql_lat = 0
            sql_raw = str(e)

        return SecondLoopResult(
            plan={},
            initial_sql=sql,
            review={"verdict": "ok", "confidence": 0.8, "skipped": True},
            revised_sql=sql,
            final_sql=sql,
            latency={
                "sqlgen_sec": sql_lat,
                "total_sec": time.perf_counter() - total_start,
            },
            debug={"mode": "direct", "sqlgen_raw": sql_raw},
        )


class HybridEngine:
    """
    Hybrid engine - routes simple queries to direct SQL, complex to full pipeline.
    """

    def __init__(
        self,
        direct_engine: DirectSQLEngine,
        full_engine: SecondLoopEngine,
        complexity_threshold: int = 4,
    ):
        self.direct_engine = direct_engine
        self.full_engine = full_engine
        self.complexity_threshold = complexity_threshold

    def _compute_complexity(self, question: str, schema_text: str) -> int:
        """Compute query complexity (0-10)."""
        import re

        score = 0
        q = question.lower()

        # Aggregation
        if any(w in q for w in ["average", "maximum", "minimum", "total", "sum"]):
            score += 2

        # Multiple conditions
        if " and " in q or " or " in q:
            score += 1

        # Subquery indicators
        if any(p in q for p in ["who has", "that have", "not have", "without", "except"]):
            score += 3

        # Ranking
        if any(w in q for w in ["highest", "lowest", "most", "least", "top", "youngest", "oldest"]):
            score += 1

        # Multiple tables
        tables_match = re.search(r"###\s*tables:\s*\[([^\]]+)\]", schema_text, re.IGNORECASE)
        if tables_match:
            tables = [t.strip() for t in tables_match.group(1).split(",") if t.strip()]
            if len(tables) > 2:
                score += 3
            elif len(tables) > 1:
                score += 1

        return min(score, 10)

    def run(self, question: str, schema_text: str, **kwargs) -> SecondLoopResult:
        """Route based on complexity."""
        complexity = self._compute_complexity(question, schema_text)

        if complexity < self.complexity_threshold:
            result = self.direct_engine.run(question, schema_text, **kwargs)
            if result.debug is None:
                result.debug = {}
            result.debug["routing"] = "direct"
            result.debug["complexity"] = complexity
        else:
            result = self.full_engine.run(question, schema_text, **kwargs)
            if result.debug is None:
                result.debug = {}
            result.debug["routing"] = "full_pipeline"
            result.debug["complexity"] = complexity

        return result
