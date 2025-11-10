from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class PlannerState:
    
    """State passed through the baseline planner.

    - question: natural language question
    - db_id: database identifier (Spider_dev db name)
    - schema_text: textual schema description for the db_id
    - plan: structured query plan produced by the planner node
    - sql: final SQL produced by the sql-generation node
    """

    question: str
    db_id: Optional[str] = None
    schema_text: Optional[str] = None
    plan: Dict[str, Any] = field(default_factory=dict)
    sql: Optional[str] = None