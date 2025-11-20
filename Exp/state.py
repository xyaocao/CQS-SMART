from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class PlannerAgentState:
   """State passed through the planner.
    - question: natural language question
    - db_id: database identifier (Spider_dev db name)
    - schema_text: textual schema description for the db_id
    - plan: structured query plan produced by the planner node
    - raw: raw output from the planner LLM
    - latency: time taken by the planner LLM
    """
   question: str
   db_id: Optional[str] = None
   schema_text: Optional[str] = None
   plan: Dict[str, Any] = field(default_factory=dict)
   raw: str = ""
   latency: float = 0.0

@dataclass
class SkepticAgentState:
    """State passed through the skeptic agent.
    - question: natural language question
    - schema_text: textual schema description for the db_id
    - plan: textual representation of the plan to critique
    - critical: critical questions to consider during critique
    - feedback: structured feedback produced by the skeptic
    - raw: raw output from the skeptic LLM
    - latency: time taken by the skeptic LLM
    """
    question: str
    schema_text: Optional[str] = None
    plan: str = ""
    critical: str = ""
    feedback: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    latency: float = 0.0

@dataclass
class ReasonerAgentState:
    """State passed through the reasoner agent.
    - question: natural language question
    - schema_text: textual schema description for the db_id
    - plan: textual representation of the plan to reason about
    - feedback: textual representation of the skeptic's feedback
    - decision: structured decision produced by the reasoner
    - decision_raw: raw output from the reasoner LLM
    - decision_latency: time taken by the reasoner LLM
    - sql: final SQL produced by the sql-generation node
    - sql_raw: raw output from the SQL generation LLM
    - sql_latency: time taken by the SQL generation LLM
    """
    question: str
    schema_text: Optional[str] = None
    plan: str = ""
    feedback: str = ""
    decision: Dict[str, Any] = field(default_factory=dict)
    decision_raw: str = ""
    decision_latency: float = 0.0
    sql: str= ""
    sql_raw: str = ""
    sql_latency: float = 0.0