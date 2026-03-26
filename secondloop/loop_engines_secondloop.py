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
import ast
import re
from difflib import get_close_matches
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False


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
        enable_planner: bool = True,  # Ablation: skip planner when False
        enable_reviewer: bool = True,  # Ablation: skip reviewer when False
        no_schema_check: bool = False,  # Ablation: disable schema consistency check
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
            enable_planner: Whether to run planner (False for ablation)
            enable_reviewer: Whether to run reviewer (False for ablation)
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
        self.enable_planner = enable_planner
        self.enable_reviewer = enable_reviewer
        self.no_schema_check = no_schema_check

    @staticmethod
    def _normalize_ident(name: str) -> str:
        return (name or "").replace("`", "").strip().lower()

    def _extract_linked_schema(self, schema_text: str) -> Dict[str, Any]:
        """
        Extract schema-linking constraints and global schema identifiers from schema text.
        Returns normalized identifiers for deterministic validation.
        """
        schema_text = schema_text or ""
        linked_tables: set[str] = set()
        linked_columns: set[str] = set()

        # Parse explicit schema-linking section if present.
        m_tables = re.search(r"###\s*tables:\s*(\[[^\]]*\])", schema_text, re.IGNORECASE)
        if m_tables:
            try:
                arr = ast.literal_eval(m_tables.group(1))
                if isinstance(arr, list):
                    linked_tables = {self._normalize_ident(x) for x in arr if isinstance(x, str) and x.strip()}
            except Exception:
                pass

        m_cols = re.search(r"###\s*columns:\s*(\[[^\]]*\])", schema_text, re.IGNORECASE)
        if m_cols:
            try:
                arr = ast.literal_eval(m_cols.group(1))
                if isinstance(arr, list):
                    for col in arr:
                        if isinstance(col, str) and "." in col:
                            parts = col.replace("`", "").split(".", 1)
                            linked_columns.add(f"{self._normalize_ident(parts[0])}.{self._normalize_ident(parts[1])}")
            except Exception:
                pass

        # Parse all available table/column identifiers from schema text DDL-like segments.
        all_tables: set[str] = set()
        all_columns: set[str] = set()
        all_column_names: set[str] = set()
        table_to_cols: Dict[str, set[str]] = {}

        for table, cols_blob in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)", schema_text):
            t_norm = self._normalize_ident(table)
            if not t_norm:
                continue
            all_tables.add(t_norm)
            table_to_cols.setdefault(t_norm, set())

            # Primary format in your schema text uses backticks.
            backtick_cols = re.findall(r"`([^`]+)`", cols_blob)
            if backtick_cols:
                col_names = [self._normalize_ident(c.split("[", 1)[0]) for c in backtick_cols]
            else:
                # Fallback for non-backtick formats.
                raw_cols = [x.strip() for x in cols_blob.split(",") if x.strip()]
                col_names = []
                for rc in raw_cols:
                    # Drop sample/value suffixes.
                    rc = rc.split("[", 1)[0].split(":", 1)[0].strip()
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rc):
                        col_names.append(self._normalize_ident(rc))

            for c_norm in col_names:
                if not c_norm:
                    continue
                all_columns.add(f"{t_norm}.{c_norm}")
                all_column_names.add(c_norm)
                table_to_cols[t_norm].add(c_norm)

        return {
            "linked_tables": linked_tables,
            "linked_columns": linked_columns,
            "linked_column_names": {c.split(".", 1)[1] for c in linked_columns if "." in c},
            "all_tables": all_tables,
            "all_columns": all_columns,
            "all_column_names": all_column_names,
            "table_to_cols": table_to_cols,
        }

    def _extract_sql_refs(self, sql: str) -> Dict[str, set[str]]:
        """Extract referenced tables/columns from SQL."""
        sql = (sql or "").strip()
        sql_tables: set[str] = set()
        sql_qualified_cols: set[str] = set()
        sql_unqualified_cols: set[str] = set()
        has_star = False

        if not sql:
            return {
                "tables": sql_tables,
                "qualified_cols": sql_qualified_cols,
                "unqualified_cols": sql_unqualified_cols,
                "has_star": has_star,
            }

        if HAS_SQLGLOT:
            try:
                parsed = sqlglot.parse_one(sql, read="sqlite")
                alias_to_table: Dict[str, str] = {}
                for t in parsed.find_all(sqlglot.exp.Table):
                    table_name = self._normalize_ident(t.name)
                    if table_name:
                        sql_tables.add(table_name)
                    alias = self._normalize_ident(t.alias_or_name)
                    if alias and table_name:
                        alias_to_table[alias] = table_name

                for c in parsed.find_all(sqlglot.exp.Column):
                    col_name = self._normalize_ident(c.alias_or_name)
                    if not col_name:
                        continue
                    tbl = self._normalize_ident(c.table)
                    if tbl:
                        real_table = alias_to_table.get(tbl, tbl)
                        sql_qualified_cols.add(f"{real_table}.{col_name}")
                    else:
                        sql_unqualified_cols.add(col_name)

                has_star = bool(list(parsed.find_all(sqlglot.exp.Star)))
                return {
                    "tables": sql_tables,
                    "qualified_cols": sql_qualified_cols,
                    "unqualified_cols": sql_unqualified_cols,
                    "has_star": has_star,
                }
            except Exception:
                pass

        # Regex fallback.
        for t in re.findall(r"(?:from|join)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", sql, flags=re.IGNORECASE):
            sql_tables.add(self._normalize_ident(t))
        for t, c in re.findall(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", sql):
            sql_qualified_cols.add(f"{self._normalize_ident(t)}.{self._normalize_ident(c)}")
        for c in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", sql):
            sql_unqualified_cols.add(self._normalize_ident(c))
        has_star = bool(re.search(r"\*", sql))
        return {
            "tables": sql_tables,
            "qualified_cols": sql_qualified_cols,
            "unqualified_cols": sql_unqualified_cols,
            "has_star": has_star,
        }

    def _schema_consistency_check(self, sql: str, schema_text: str) -> Dict[str, Any]:
        """
        Deterministic checker: verifies SQL identifiers are consistent with schema text.
        Hard fail: SQL references table/column that does not exist in schema text.
        Soft fail: SQL deviates from schema-linking constraints.
        """
        schema_info = self._extract_linked_schema(schema_text)
        sql_refs = self._extract_sql_refs(sql)

        all_tables = schema_info["all_tables"]
        all_columns = schema_info["all_columns"]
        all_column_names = schema_info["all_column_names"]
        linked_tables = schema_info["linked_tables"]
        linked_columns = schema_info["linked_columns"]
        linked_column_names = schema_info["linked_column_names"]
        table_to_cols = schema_info["table_to_cols"]

        issues: List[str] = []
        hints: List[str] = []

        unknown_tables = sorted(t for t in sql_refs["tables"] if t and all_tables and t not in all_tables)
        unknown_qualified_cols = sorted(c for c in sql_refs["qualified_cols"] if c and all_columns and c not in all_columns)
        unknown_unqualified_cols = sorted(c for c in sql_refs["unqualified_cols"] if c and all_column_names and c not in all_column_names)

        hard_fail = bool(unknown_tables or unknown_qualified_cols or unknown_unqualified_cols)
        if unknown_tables:
            issues.append(f"SQL references table(s) not in schema: {unknown_tables}")
            hints.append("Use only tables that appear in schema text.")
        for qc in unknown_qualified_cols:
            t, c = qc.split(".", 1)
            issues.append(f"SQL references unknown column: {qc}")
            candidates = sorted(table_to_cols.get(t, set()))
            if candidates:
                close = get_close_matches(c, candidates, n=3, cutoff=0.5)
                if close:
                    hints.append(f"For table '{t}', replace '{c}' with one of: {close}")
        if unknown_unqualified_cols:
            issues.append(f"SQL references unknown unqualified column(s): {unknown_unqualified_cols}")
            close_any = []
            for c in unknown_unqualified_cols:
                m = get_close_matches(c, sorted(all_column_names), n=3, cutoff=0.5)
                if m:
                    close_any.append((c, m))
            for c, m in close_any:
                hints.append(f"Replace '{c}' with schema column candidates: {m}")

        # Soft constraints based on schema-linking section.
        soft_fail = False
        extra_tables = []
        if linked_tables:
            extra_tables = sorted(t for t in sql_refs["tables"] if t not in linked_tables)
            if extra_tables:
                soft_fail = True
                issues.append(f"SQL uses table(s) outside schema-linking results: {extra_tables}")
                hints.append(f"Prefer linked tables: {sorted(linked_tables)}")

        if linked_columns and sql_refs["qualified_cols"]:
            extra_qcols = sorted(c for c in sql_refs["qualified_cols"] if c not in linked_columns)
            if extra_qcols:
                soft_fail = True
                issues.append(f"SQL uses column(s) outside linked columns: {extra_qcols[:8]}")
                hints.append("Use `### columns` from schema linking as the primary column set.")

        if linked_column_names and sql_refs["unqualified_cols"]:
            extra_ucols = sorted(c for c in sql_refs["unqualified_cols"] if c not in linked_column_names and c not in all_column_names)
            if extra_ucols:
                soft_fail = True
                issues.append(f"SQL uses suspicious unqualified column(s): {extra_ucols[:8]}")
                hints.append("Qualify columns with table aliases to avoid ambiguous or hallucinated names.")

        return {
            "hard_fail": hard_fail,
            "soft_fail": soft_fail,
            "issues": issues,
            "hints": hints,
            "unknown_tables": unknown_tables,
            "unknown_qualified_cols": unknown_qualified_cols,
            "unknown_unqualified_cols": unknown_unqualified_cols,
        }

    def _safe_execute(self, db_path: Optional[str], sql: str) -> tuple[bool, Optional[str]]:
        """Execute SQL safely and return (ok, error_message)."""
        if not self.db_executor or not db_path:
            return False, "db_executor unavailable"
        if not sql or not sql.strip():
            return False, "empty SQL"
        try:
            _, err = self.db_executor(db_path, sql)
            if err:
                return False, str(err)
            return True, None
        except Exception as e:
            return False, str(e)

    def _pick_first_executable(self, candidates: List[tuple[str, str]], db_path: Optional[str], schema_text: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """
        Pick the first executable SQL from labeled candidates.
        candidates: [(label, sql), ...]
        Returns: (sql, label) or (None, None)
        """
        seen = set()
        best_exec = None
        for label, sql in candidates:
            normalized = " ".join((sql or "").split()).lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)

            # Prefer schema-consistent candidates first.
            if schema_text:
                check = self._schema_consistency_check(sql, schema_text)
                if check["hard_fail"]:
                    continue

            ok, _ = self._safe_execute(db_path, sql)
            if ok:
                return sql, label
            if best_exec is None:
                best_exec = (sql, label)
        # Fallback: return first executable candidate even if consistency is uncertain.
        if best_exec:
            return best_exec
        return None, None

    def run(self, question: str, schema_text: str, db_path: str = None, **kwargs) -> SecondLoopResult:
        """Run the simplified SecondLoop pipeline."""
        total_start = time.perf_counter()
        latency = {}
        debug = {}
        voting_info = None

        # === STAGE 1: PLANNER ===
        if self.enable_planner:
            try:
                plan, plan_lat, plan_raw = self.planner.run(question, schema_text)
                latency["planner_sec"] = plan_lat
                debug["planner_raw"] = plan_raw
                debug["join_reasoning"] = plan.get("join_reasoning", "")
            except Exception as e:
                plan = {"tables": [], "columns": [], "error": str(e)}
                latency["planner_sec"] = 0
                debug["planner_error"] = str(e)
        else:
            plan = {}
            latency["planner_sec"] = 0
            debug["planner_skipped"] = True

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
        if self.enable_reviewer:
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

            # Deterministic consistency checker against schema text (high-ROI guardrail).
            if not self.no_schema_check:
                schema_check_initial = self._schema_consistency_check(initial_sql, schema_text)
                debug["schema_check_initial"] = schema_check_initial
                if schema_check_initial["hard_fail"] or schema_check_initial["soft_fail"]:
                    review = dict(review) if isinstance(review, dict) else {}
                    review["verdict"] = "needs_revision"
                    review["confidence"] = min(float(review.get("confidence", 0.6)), 0.4)
                    existing_issues = review.get("issues", [])
                    existing_hints = review.get("revision_hints", [])
                    if not isinstance(existing_issues, list):
                        existing_issues = [str(existing_issues)]
                    if not isinstance(existing_hints, list):
                        existing_hints = [str(existing_hints)]
                    review["issues"] = existing_issues + schema_check_initial["issues"]
                    review["revision_hints"] = existing_hints + schema_check_initial["hints"]
            else:
                debug["schema_check_initial"] = {"skipped": True, "no_schema_check": True}

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
                        # Deduplicate while preserving order for clearer feedback.
                        seen_issue = set()
                        issues = [i for i in issues if not (i in seen_issue or seen_issue.add(i))]
                        seen_hint = set()
                        hints = [h for h in hints if not (h in seen_hint or seen_hint.add(h))]

                        revised_sql, regen_lat, regen_raw = self.sql_gen_with_feedback.run(
                            question, schema_text, initial_sql, issues, hints
                        )
                        latency["regeneration_sec"] = regen_lat
                        debug["regeneration_raw"] = regen_raw
                        revision_applied = (revised_sql != initial_sql)

                        # If regenerated SQL is still schema-inconsistent, do one focused retry.
                        if not self.no_schema_check:
                            schema_check_revised = self._schema_consistency_check(revised_sql, schema_text)
                            debug["schema_check_revised"] = schema_check_revised
                            if schema_check_revised["hard_fail"] and self.sql_gen_with_feedback:
                                retry_issues = ["Regenerated SQL still violates schema consistency checks."] + schema_check_revised["issues"]
                                retry_hints = schema_check_revised["hints"] + [
                                    "Use exact table/column names present in schema text.",
                                    "Do not invent new columns or table names.",
                                ]
                                revised_sql_2, regen_lat_2, regen_raw_2 = self.sql_gen_with_feedback.run(
                                    question, schema_text, revised_sql, retry_issues, retry_hints
                                )
                                latency["regeneration_retry_sec"] = regen_lat_2
                                debug["regeneration_retry_raw"] = regen_raw_2
                                if revised_sql_2 and revised_sql_2 != revised_sql:
                                    revised_sql = revised_sql_2
                                    revision_applied = True
                        else:
                            debug["schema_check_revised"] = {"skipped": True, "no_schema_check": True}
                    except Exception as e:
                        latency["regeneration_sec"] = 0
                        debug["regeneration_error"] = str(e)
        else:
            # Reviewer skipped (ablation)
            review = {"verdict": "ok", "confidence": 1.0, "skipped": True}
            latency["reviewer_sec"] = 0
            debug["reviewer_skipped"] = True
            revised_sql = initial_sql
            revision_applied = False
            needs_regeneration = False

        final_sql = revised_sql
        refiner_applied = False
        verification_applied = False
        verification_result = None

        # === STAGE 4.5: SELF-VERIFICATION (Optional) ===
        # Run verification if enabled AND reviewer passed with "ok"
        # Skip if reviewer was disabled (nothing to verify against)
        if (self.enable_verification and self.verifier and
            self.enable_reviewer and
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

        # If a regenerated/verified SQL regressed into execution error but an
        # earlier candidate is executable, fall back before refinement.
        if self.db_executor and db_path:
            fallback_sql, fallback_label = self._pick_first_executable(
                [
                    ("final_sql", final_sql),
                    ("revised_sql", revised_sql),
                    ("initial_sql", initial_sql),
                ],
                db_path=db_path,
                schema_text=schema_text,
            )
            if fallback_sql and fallback_sql != final_sql:
                debug["execution_fallback_before_refiner"] = {
                    "from": "final_sql",
                    "to": fallback_label,
                }
                final_sql = fallback_sql

        # Final schema gate before refiner: do not let hard-invalid SQL proceed.
        if not self.no_schema_check:
            final_schema_check = self._schema_consistency_check(final_sql, schema_text)
            debug["schema_check_final_pre_refiner"] = final_schema_check
            if final_schema_check["hard_fail"]:
                # Try to recover with earlier candidates that pass both checks.
                fallback_sql, fallback_label = self._pick_first_executable(
                    [
                        ("revised_sql", revised_sql),
                        ("initial_sql", initial_sql),
                    ],
                    db_path=db_path,
                    schema_text=schema_text,
                )
                if fallback_sql:
                    debug["schema_gate_fallback"] = {
                        "from": "final_sql",
                        "to": fallback_label,
                    }
                    final_sql = fallback_sql
        else:
            debug["schema_check_final_pre_refiner"] = {"skipped": True, "no_schema_check": True}

        # === STAGE 5: EXECUTION + REFINER (Optional) ===
        if self.enable_refiner and self.refiner and db_path and self.db_executor:
            try:
                exec_result, exec_error = self.db_executor(db_path, final_sql)

                if exec_error:
                    debug["execution_error"] = exec_error
                    current_sql = final_sql
                    current_error = exec_error
                    refiner_lat_total = 0.0
                    max_refiner_attempts = 2

                    for attempt in range(max_refiner_attempts):
                        try:
                            fixed_sql, analysis, refine_lat, refine_raw = self.refiner.run(
                                question, schema_text, current_sql, current_error
                            )
                            refiner_lat_total += refine_lat
                            debug[f"refiner_raw_attempt_{attempt + 1}"] = refine_raw
                            debug[f"refiner_analysis_attempt_{attempt + 1}"] = analysis

                            # No new SQL generated; stop retrying.
                            if not fixed_sql or fixed_sql.strip() == current_sql.strip():
                                break

                            ok, new_error = self._safe_execute(db_path, fixed_sql)
                            if ok:
                                final_sql = fixed_sql
                                refiner_applied = True
                                current_error = None
                                break

                            # Keep iterating on the newly generated failing SQL.
                            current_sql = fixed_sql
                            current_error = new_error or current_error
                            debug[f"refiner_exec_error_attempt_{attempt + 1}"] = current_error
                        except Exception as e:
                            debug[f"refiner_error_attempt_{attempt + 1}"] = str(e)
                            break

                    if refiner_lat_total > 0:
                        latency["refiner_sec"] = refiner_lat_total

                    # Final safety fallback: prefer any executable earlier candidate.
                    if current_error:
                        fallback_sql, fallback_label = self._pick_first_executable(
                            [
                                ("final_sql_after_refiner", final_sql),
                                ("revised_sql", revised_sql),
                                ("initial_sql", initial_sql),
                            ],
                            db_path=db_path,
                            schema_text=schema_text,
                        )
                        if fallback_sql and fallback_sql != final_sql:
                            debug["execution_fallback_after_refiner"] = {
                                "from": "final_sql_after_refiner",
                                "to": fallback_label,
                            }
                            final_sql = fallback_sql
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
