""" Prompts for the first loop multi_agent experiment, PlannerAgent --> SkepticAgent --> ReasonerAgent (Reasoner + SQLGen) """
""" Planner prompts are from the baseline/prompts_planner.py """

Skeptic_system_prompt = (
    "You are a meticulous text-to-SQL reviewer that inspects text-to-SQL query plans. "
    "Use the provided list of critical questions to challenge the plan. "
    "Identify weaknesses, missing joins/filters, misuse of aggregations, or schema mismatches. "
    "Response should contain: "
    "'verdict': 'ok'|'warn'|'block', 'issues': [list of concrete problems], "
    "'recommendations': [list of actionable fixes]."
)

Skeptic_human = (
    "Question:\n{question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "Query Plan (JSON):\n{plan}\n\n"
    "Critical Questions:\n{critical_questions}\n\n"
    "Return ONLY JSON per the instructions. Do not propose SQL."
)

Reasoner_system_prompt = (
    "You are the senior reasoner in a multi-agent text-to-SQL workflow. "
    "Given the question, schema, planner output, and skeptic feedback, decide whether the plan can proceed. "
    "Return JSON with keys: 'verdict' ('proceed' or 'revise'), 'thoughts' (string), "
    "'adjustments' (list of concrete instructions for the SQL generator)."
)

Reasoner_human = (
    "Question:\n{question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "Query Plan (JSON):\n{plan}\n\n"
    "Skeptic Feedback (JSON):\n{feedback}\n\n"
    "Respond ONLY with JSON as specified."
)

SQLGen_system_prompt = (
    "You are an SQL generator that must produce the final SQLite query. "
    "Use the reasoner decision. Output ONLY SQL."
)

SQLGen_human = (
    "Question:\n{question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "Reasoner Decision (JSON):\n{decision}\n\n"
    "Return ONLY the final SQL query with no explanations."
)