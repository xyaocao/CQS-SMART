import time
from dataclasses import dataclass
from typing import Any, Dict

class LoopEngine:
    """Abstract base class for loop engines."""
    def run(self, question: str, schema_text: str) -> 'LoopResult':
        raise NotImplementedError()
    
@dataclass
class LoopResult:
    plan: Dict[str, Any]
    skeptic_feedback: Dict[str, Any]
    reasoner_decision: Dict[str, Any]
    final_sql: str
    latency: Dict[str, float]

@dataclass
class LoopStageError(RuntimeError):
    stage: str
    original: Exception
    elapsed: float
    partial: Dict[str, Any]

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, f"{self.stage} failure: {self.original}")

class FirstLoopEngine:
    """Planner → Skeptic → Reasoner(+SQL) executor that records per-stage latencies."""
    def __init__(self, planner, skeptic, reasoner) -> None:
        self.planner = planner
        self.skeptic = skeptic
        self.reasoner = reasoner

    def run(self, question: str, schema_text: str) -> LoopResult:
        total_start = time.perf_counter()
        result: Dict[str, Any] = {"latency": {}}

        try:
            plan, plan_latency = self.planner.run(question, schema_text)
            result["plan"] = plan
            result["latency"]["planner_sec"] = plan_latency
        except Exception as exc:  # pragma: no cover - depends on remote LLMs
            raise LoopStageError("Planner", exc, time.perf_counter() - total_start, result) from exc

        try:
            feedback, skeptic_latency = self.skeptic.run(question, schema_text, plan)
            result["skeptic_feedback"] = feedback
            result["latency"]["skeptic_sec"] = skeptic_latency
        except Exception as exc:  
            raise LoopStageError("Skeptic", exc, time.perf_counter() - total_start, result) from exc

        try:
            decision, decision_latency, final_sql, sql_latency = self.reasoner.run(
                question, schema_text, plan, feedback
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
            skeptic_feedback=result["skeptic_feedback"],
            reasoner_decision=result["reasoner_decision"],
            final_sql=result["final_sql"],
            latency=result["latency"],
        )


