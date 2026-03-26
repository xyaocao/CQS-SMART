import time
from dataclasses import dataclass
from typing import Any, Dict, List

class LoopEngine:
    """Abstract base class for loop engines."""

    def run(self, question: str, schema_text: str, **kwargs: Any) -> "LoopResult":
        """
        Execute one or more planner/skeptic/reasoner passes for a question.
        """
        raise NotImplementedError()

@dataclass
class LoopResult:
    plan: Dict[str, Any]
    skeptic_feedback: Dict[str, Any]
    reasoner_decision: Dict[str, Any]
    final_sql: str
    latency: Dict[str, float]
    debug: Dict[str, Any] | None = None
    # For heteroMAD: track which LLM was chosen for each agent call
    llm_choices: Dict[str, str] | None = None

@dataclass
class LoopStageError(RuntimeError):
    stage: str
    original: Exception
    elapsed: float
    partial: Dict[str, Any]

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, f"{self.stage} failure: {self.original}")

class FirstLoopEngine(LoopEngine):
    """Single pass: Planner -> Skeptic -> Reasoner(+SQL) with per-stage latencies."""

    def __init__(self, planner, skeptic, reasoner):
        self.planner = planner
        self.skeptic = skeptic
        self.reasoner = reasoner

    def run(self, question: str, schema_text: str, **_: Any) -> LoopResult:
        """
        Run one planner -> skeptic -> reasoner(+SQL) pass.
        """
        total_start = time.perf_counter()
        result: Dict[str, Any] = {"latency": {}}

        try:
            plan, plan_latency = self.planner.run(question, schema_text)
            result["plan"] = plan
            result["latency"]["planner_sec"] = plan_latency
        except Exception as exc:
            raise LoopStageError("Planner", exc, time.perf_counter() - total_start, result) from exc

        try:
            feedback, skeptic_latency = self.skeptic.run(question, schema_text, plan)
            result["skeptic_feedback"] = feedback
            result["latency"]["skeptic_sec"] = skeptic_latency
        except Exception as exc:  
            raise LoopStageError("Skeptic", exc, time.perf_counter() - total_start, result) from exc

        try:
            decision, decision_latency, final_sql, sql_latency = self.reasoner.run(question, schema_text, plan, feedback)
            result["reasoner_decision"] = decision
            result["final_sql"] = final_sql
            result["latency"]["reasoner_decision_sec"] = decision_latency
            result["latency"]["sqlgen_sec"] = sql_latency
        except Exception as exc:
            raise LoopStageError("Reasoner/SQL", exc, time.perf_counter() - total_start, result) from exc

        # result["latency"]["execution_sec"] = 0.0
        result["latency"]["total_sec"] = time.perf_counter() - total_start
        return LoopResult(
            plan=result["plan"],
            skeptic_feedback=result["skeptic_feedback"],
            reasoner_decision=result["reasoner_decision"],
            final_sql=result["final_sql"],
            latency=result["latency"],
            debug=None,
        )


class BaseMADEngine(LoopEngine):
    """Base multi-agent debate (BaseMAD) engine.

    Pipeline:
      1) Planner produces initial plan s0.
      2) Defender and Skeptic debate for up to max_debate_rounds.
         - Round 1: Both independently evaluate the plan in parallel.
           * If viewpoints match (both support or both want similar revisions) → consensus reached (counts as 1 round).
           * If viewpoints don't match → continue to sequential debate.
         - Round 2+: Sequential debate (Defender refines → Skeptic evaluates).
           * If consensus is reached, adopt the debated viewpoint.
           * If views stabilize (no meaningful change), terminate early.
      3) Reasoner consumes the plan and debate summary and produces final SQL.
    """

    def __init__(self, planner, defender, skeptic, reasoner, max_debate_rounds: int = 3, enable_early_termination: bool = True, min_debate_rounds: int = 2,):
        self.planner = planner
        self.defender = defender
        self.skeptic = skeptic
        self.reasoner = reasoner
        self.max_debate_rounds = max(1, int(max_debate_rounds))
        self.enable_early_termination = bool(enable_early_termination)
        self.min_debate_rounds = max(1, int(min_debate_rounds))

    def viewpoints_match(self, defense: Dict[str, Any], skeptic: Dict[str, Any]) -> bool:
        """Check if defender and skeptic viewpoints match (both support or both want similar revisions).
        
        Returns True if:
        - Both support the plan (defender stance='support_plan' AND skeptic position='accept')
        - OR both want revisions and have similar recommendations
        """
        defender_stance = defense.get("stance", "").lower()
        skeptic_position = skeptic.get("position", "").lower()
        skeptic_consensus = skeptic.get("consensus", False)
        
        # Both support: defender supports plan AND skeptic accepts
        if defender_stance == "support_plan" and skeptic_position == "accept" and skeptic_consensus:
            return True
        
        # Both want revisions: check if they have similar concerns
        if defender_stance == "revise_plan" and skeptic_position == "reject":
            # Check if they identify similar issues
            defender_notes = defense.get("notes_for_reasoner", [])
            skeptic_issues = skeptic.get("issues", [])
            skeptic_recs = skeptic.get("recommendations", [])
            
            # If skeptic has no issues or defender acknowledges the same issues, consider it a match
            if not skeptic_issues or (defender_notes and len(defender_notes) > 0):
                # Check if defender's notes address skeptic's recommendations
                if skeptic_recs:
                    defender_notes_str = " ".join(str(note).lower() for note in defender_notes)
                    skeptic_recs_str = " ".join(str(rec).lower() for rec in skeptic_recs[:2])  # Check first 2 recommendations
                    # If defender notes mention key terms from skeptic recommendations, consider it a match
                    if any(keyword in defender_notes_str for keyword in skeptic_recs_str.split()[:5] if len(keyword) > 4):
                        return True
                # If skeptic has no critical issues and defender wants to revise, they might be aligned
                if not skeptic_issues:
                    return True
        
        return False

    def views_stable(self, prev_defense: Dict[str, Any], curr_defense: Dict[str, Any], prev_skeptic: Dict[str, Any], curr_skeptic: Dict[str, Any]) -> bool:
        """Check if defender and skeptic views have stabilized (no meaningful change)."""
        if not self.enable_early_termination:
            return False
        
        # Compare key fields that indicate meaningful change
        def get_viewpoint_key(d: Dict[str, Any]) -> str:
            # Extract a normalized key from viewpoint/position/stance
            viewpoint = str(d.get("viewpoint", d.get("position", d.get("stance", "")))).lower().strip()
            # Also consider issues/recommendations for skeptic
            issues = str(d.get("issues", [])).lower()
            return f"{viewpoint}|{issues}"
        
        prev_def_key = get_viewpoint_key(prev_defense)
        curr_def_key = get_viewpoint_key(curr_defense)
        prev_skept_key = get_viewpoint_key(prev_skeptic)
        curr_skept_key = get_viewpoint_key(curr_skeptic)
        
        # Views are stable if both defender and skeptic haven't changed meaningfully
        def_stable = prev_def_key == curr_def_key
        skept_stable = prev_skept_key == curr_skept_key
        
        return def_stable and skept_stable

    @staticmethod
    def dedup_list(items: Any, limit: int = 5) -> List[Any]:
        """Deduplicate while preserving order; drop empty; keep first `limit`."""
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

    def run(self, question: str, schema_text: str, **_: Any) -> "LoopResult":
        total_start = time.perf_counter()
        result: Dict[str, Any] = {"latency": {}}
        debug: Dict[str, Any] = {"debate_rounds": [], "max_debate_rounds": self.max_debate_rounds}

        # 1) Planner: initial plan s0
        try:
            plan, plan_latency = self.planner.run(question, schema_text)
            result["plan"] = plan
            result["latency"]["planner_sec"] = plan_latency
        except Exception as exc:
            raise LoopStageError("Planner", exc, time.perf_counter() - total_start, result) from exc

        # 2) Debate between Defender and Skeptic
        last_defense: Dict[str, Any] = {}
        last_skeptic: Dict[str, Any] = {}
        prev_defense: Dict[str, Any] = {}
        prev_skeptic: Dict[str, Any] = {}
        consensus_reached = False
        early_terminated = False
        termination_reason = None
        defender_total = 0.0
        skeptic_total = 0.0

        # Round 1: Independent parallel evaluation
        round_idx = 1
        try:
            # Defender independently evaluates the plan (no skeptic feedback yet)
            defense, defender_latency = self.defender.run(
                question, schema_text, plan, skeptic_feedback=None
            )
            defense = defense.copy()
            if isinstance(defense.get("justification"), list):
                defense["justification"] = self.dedup_list(defense.get("justification"))
            if isinstance(defense.get("notes_for_reasoner"), list):
                defense["notes_for_reasoner"] = self.dedup_list(defense.get("notes_for_reasoner"))
            defender_total += defender_latency
            last_defense = defense

            # Skeptic independently evaluates the plan (no defender feedback yet)
            skeptic_feedback, skeptic_latency = self.skeptic.run(
                question, schema_text, plan, execution_feedback=None
            )
            skeptic_feedback = skeptic_feedback.copy()
            if isinstance(skeptic_feedback.get("issues"), list):
                skeptic_feedback["issues"] = self.dedup_list(skeptic_feedback.get("issues"))
            if isinstance(skeptic_feedback.get("recommendations"), list):
                skeptic_feedback["recommendations"] = self.dedup_list(skeptic_feedback.get("recommendations"))
            skeptic_total += skeptic_latency
            last_skeptic = skeptic_feedback

            # Check if viewpoints match (both support or both want similar revisions)
            viewpoints_match = self.viewpoints_match(defense, skeptic_feedback)
            
            # If viewpoints match, set consensus flag in skeptic feedback for consistency
            if viewpoints_match:
                skeptic_feedback["consensus"] = True
                if skeptic_feedback.get("position", "").lower() != "accept":
                    skeptic_feedback["position"] = "accept"
                last_skeptic = skeptic_feedback
            
            debug["debate_rounds"].append(
                {
                    "round": round_idx,
                    "defender_view": defense,
                    "skeptic_view": last_skeptic,
                    "consensus": viewpoints_match,
                    "defender_latency_sec": defender_latency,
                    "skeptic_latency_sec": skeptic_latency,
                    "evaluation_mode": "parallel_independent"
                }
            )

            # If viewpoints match, consensus reached in round 1
            if viewpoints_match:
                consensus_reached = True
                termination_reason = "consensus_reached_parallel_evaluation"
            else:
                # Continue with sequential debate starting from round 2
                prev_defense = last_defense.copy()
                prev_skeptic = last_skeptic.copy()
                round_idx += 1

        except Exception as exc:
            raise LoopStageError(
                f"Debate_round_{round_idx}", exc,
                time.perf_counter() - total_start,
                result,
            ) from exc

        # Sequential debate rounds (if viewpoints didn't match in round 1)
        if not consensus_reached:
            for round_idx in range(round_idx, self.max_debate_rounds + 1):
                # Defender refines the plan given the last skeptic attack
                try:
                    defense, defender_latency = self.defender.run(
                        question, schema_text, plan, skeptic_feedback=last_skeptic
                    )
                    defense = defense.copy()
                    if isinstance(defense.get("justification"), list):
                        defense["justification"] = self.dedup_list(defense.get("justification"))
                    if isinstance(defense.get("notes_for_reasoner"), list):
                        defense["notes_for_reasoner"] = self.dedup_list(defense.get("notes_for_reasoner"))
                    defender_total += defender_latency
                    prev_defense = last_defense.copy()
                    last_defense = defense
                except Exception as exc:
                    raise LoopStageError(
                        f"Defender_round_{round_idx}", exc,
                        time.perf_counter() - total_start,
                        result,
                    ) from exc

                # Skeptic attacks the defender viewpoint (passed as execution_feedback)
                try:
                    skeptic_feedback, skeptic_latency = self.skeptic.run(
                        question, schema_text, plan, execution_feedback=defense
                    )
                    skeptic_feedback = skeptic_feedback.copy()
                    if isinstance(skeptic_feedback.get("issues"), list):
                        skeptic_feedback["issues"] = self.dedup_list(skeptic_feedback.get("issues"))
                    if isinstance(skeptic_feedback.get("recommendations"), list):
                        skeptic_feedback["recommendations"] = self.dedup_list(skeptic_feedback.get("recommendations"))
                    skeptic_total += skeptic_latency
                    prev_skeptic = last_skeptic.copy()
                    last_skeptic = skeptic_feedback
                except Exception as exc:
                    raise LoopStageError(
                        f"Skeptic_round_{round_idx}", exc,
                        time.perf_counter() - total_start,
                        result,
                    ) from exc

                consensus_flag = bool(last_skeptic.get("consensus", False))
                debug["debate_rounds"].append(
                    {
                        "round": round_idx,
                        "defender_view": defense,
                        "skeptic_view": last_skeptic,
                        "consensus": consensus_flag,
                        "defender_latency_sec": defender_latency,
                        "skeptic_latency_sec": skeptic_latency,
                        "evaluation_mode": "sequential",
                    }
                )

                # Early termination checks (only after minimum rounds)
                if round_idx >= self.min_debate_rounds:
                    # Check for consensus first
                    if consensus_flag:
                        consensus_reached = True
                        termination_reason = "consensus_reached"
                        break

                    # Check for stability (views not changing)
                    if (
                        round_idx > self.min_debate_rounds
                        and self.views_stable(prev_defense, last_defense, prev_skeptic, last_skeptic)
                    ):
                        early_terminated = True
                        termination_reason = "views_stabilized"
                        break
                elif consensus_flag:
                    # Consensus can happen even before min_rounds
                    consensus_reached = True
                    termination_reason = "consensus_reached_early"
                    break

        result["latency"]["defender_total_sec"] = defender_total
        result["latency"]["skeptic_total_sec"] = skeptic_total
        result["latency"]["debate_total_sec"] = defender_total + skeptic_total
        debug["consensus_reached"] = consensus_reached
        debug["early_terminated"] = early_terminated
        debug["termination_reason"] = termination_reason
        debug["actual_rounds"] = len(debug["debate_rounds"])

        # Build debate summary for the reasoner - extract key insights
        # Extract critical issues and recommendations from the debate
        all_issues = []
        all_recommendations = []
        critical_verdicts = []
        
        for round_data in debug["debate_rounds"]:
            skeptic_view = round_data.get("skeptic_view", {})
            defender_view = round_data.get("defender_view", {})
            
            # Collect issues from skeptic
            if isinstance(skeptic_view.get("issues"), list):
                all_issues.extend(skeptic_view["issues"])
            elif isinstance(skeptic_view.get("issues"), str):
                all_issues.append(skeptic_view["issues"])
            
            # Collect recommendations from both
            if isinstance(skeptic_view.get("recommendations"), list):
                all_recommendations.extend(skeptic_view["recommendations"])
            if isinstance(defender_view.get("notes_for_reasoner"), list):
                all_recommendations.extend(defender_view["notes_for_reasoner"])
            
            # Track critical verdicts
            verdict = skeptic_view.get("verdict")
            if verdict:
                critical_verdicts.append(verdict)
        
        # Deduplicate while preserving order
        seen_issues = set()
        unique_issues = []
        for issue in all_issues:
            issue_str = str(issue).lower().strip()
            if issue_str and issue_str not in seen_issues:
                seen_issues.add(issue_str)
                unique_issues.append(issue)
        
        seen_recs = set()
        unique_recommendations = []
        for rec in all_recommendations:
            rec_str = str(rec).lower().strip()
            if rec_str and rec_str not in seen_recs:
                seen_recs.add(rec_str)
                unique_recommendations.append(rec)
        
        # Build condensed debate summary for the reasoner
        debate_summary: Dict[str, Any] = {
            "consensus_reached": consensus_reached,
            "final_defender_viewpoint": last_defense.get("viewpoint", ""),
            "final_defender_stance": last_defense.get("stance", ""),
            "final_skeptic_position": last_skeptic.get("position", ""),
            "final_skeptic_verdict": last_skeptic.get("verdict", ""),
            # Condensed key insights
            "critical_issues": unique_issues[:10],  # Limit to top 10 most important
            "key_recommendations": unique_recommendations[:10],  # Limit to top 10
            "has_structural_issues": any("redundant" in str(i).lower() or "extra_table" in str(i).lower() or "unnecessary" in str(i).lower() 
                                         for i in unique_issues),
            "debate_rounds_count": len(debug["debate_rounds"]),
            # Keep full views for reference but in a structured way
            "final_defender_full": last_defense,
            "final_skeptic_full": last_skeptic,
        }

        # 3) Reasoner + SQLGen
        try:
            decision, decision_latency, final_sql, sql_latency = self.reasoner.run(
                question, schema_text, plan, feedback=debate_summary
            )
            result["reasoner_decision"] = decision
            result["final_sql"] = final_sql
            result["latency"]["reasoner_decision_sec"] = decision_latency
            result["latency"]["sqlgen_sec"] = sql_latency
        except Exception as exc:
            raise LoopStageError("Reasoner/SQL", exc, time.perf_counter() - total_start, result) from exc

        result["latency"]["execution_sec"] = 0.0
        result["latency"]["total_sec"] = time.perf_counter() - total_start

        return LoopResult(
            plan=result["plan"],
            skeptic_feedback=debate_summary,
            reasoner_decision=result["reasoner_decision"],
            final_sql=result["final_sql"],
            latency=result["latency"],
            debug=debug,
        )


class HeteroMADEngine(BaseMADEngine):
    """Heterogeneous Multi-Agent Debate (HeteroMAD) engine.
    This is identical to BaseMADEngine, but each agent randomly selects an LLM
    from a pool of candidate LLMs each time it's called. The debate logic and
    flow remain the same as BaseMAD.
    
    The LLM selection happens at the agent level, so this engine just inherits
    from BaseMADEngine and uses agents that have been initialized with LLM pools.
    """
    
    def __init__(self, planner, defender, skeptic, reasoner, max_debate_rounds: int = 3, enable_early_termination: bool = True, min_debate_rounds: int = 2):
        # HeteroMAD uses the same debate logic as BaseMAD
        # The only difference is that agents are initialized with LLM pools and select randomly on each call
        super().__init__(
            planner, defender, skeptic, reasoner,
            max_debate_rounds=max_debate_rounds,
            enable_early_termination=enable_early_termination,
            min_debate_rounds=min_debate_rounds,
        )

