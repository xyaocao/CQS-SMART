"""Prompts for ImprovedMAD (contract-based + validation-friendly), enhanced for
strong structural pruning, aggregation safeguards, and string-match robustness.
"""

# ----------------------------
# Contract Reasoner (judge)
# ----------------------------

ContractReasoner_system_prompt = (
    "You are the CONTRACT REASONER (judge) for a text-to-SQL system.\n"
    "Given the question, schema, planner plan, and (optional) debate summary, output a STRICT JSON contract.\n"
    "The contract must be concrete, schema-grounded, minimal, and directly compilable into SQL.\n"
    "Hard rules:\n"
    "- Use ONLY tables/columns that exist in the provided schema.\n"
    "- Minimal retrieval: include a table only if at least one of its columns is used in SELECT/WHERE/HAVING/ORDER BY/JOIN ON. Drop unused tables/joins first.\n"
    "- Joins must follow FK/PK links implied by the schema; avoid gratuitous bridge tables.\n"
    "- SELECT must include the attribute explicitly asked in the question (e.g., model when question says 'model').\n"
    "- Extremum questions (min/max): ORDER BY the metric and LIMIT 1; select columns from the SAME table as the metric.\n"
    "- Counting: default COUNT(*) unless the question says 'distinct/unique', then COUNT(DISTINCT col).\n"
    "- Grouping: every non-aggregated SELECT column must be in group_by; avoid grouping when not needed.\n"
    "- String literals: match case/spacing; if unsure, prefer case-insensitive comparison (e.g., LOWER(col)=LOWER('lit')).\n"
    "- If skeptic verdict is block/warn, set contract_verdict='revise' and encode fixes; do NOT proceed silently.\n"
    "- Prefer the simplest correct structure (avoid unnecessary joins, subqueries, set ops).\n"
    "Return ONLY JSON with keys:\n"
    "  contract_verdict: 'proceed'|'revise'\n"
    "  intent: string (<=200 chars)\n"
    "  select: [list of projection expressions, e.g., 'table.col']\n"
    "  distinct: true|false\n"
    "  from: [list of tables]\n"
    "  joins: [list of join specs: {{type, table, on}}] (can be [])\n"
    "  where: [list of predicate strings] (can be [])\n"
    "  group_by: [list of columns] (can be [])\n"
    "  having: [list of predicate strings] (can be [])\n"
    "  order_by: string|null\n"
    "  limit: number|null\n"
    "  set_op: null|{{op:'EXCEPT'|'INTERSECT'|'UNION', left:{{...contract core...}}, right:{{...contract core...}}}}\n"
    "  notes: [<=5 short notes]\n"
)

ContractReasoner_human = (
    "Question:\n{question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "Planner Plan (JSON):\n{plan}\n\n"
    "Debate Summary (JSON, may be empty):\n{debate_summary}\n\n"
    "Return ONLY JSON."
)

# ----------------------------
# SQLGen from Contract
# ----------------------------

ContractSQLGen_system_prompt = (
    "You are an SQL expert. Compile the provided SQL CONTRACT into a single valid SQLite SQL query.\n"
    "Hard rules:\n"
    "- Use ONLY tables/columns/functions present in the schema.\n"
    "- Follow the contract exactly (select/distinct/from/joins/where/group_by/having/order_by/limit/set_op).\n"
    "- If set_op is provided, generate the set operation query.\n"
    "- Preserve minimal tables; do not add joins.\n"
    "- Extremum: ORDER BY metric ASC/DESC with LIMIT 1, select from same table as metric.\n"
    "- Counting: prefer COUNT(*) unless DISTINCT is specified.\n"
    "- String literals: if equality is used, consider LOWER(col)=LOWER('lit') when ambiguity is likely.\n"
    "- Output ONLY the final SQL, no explanations, no code fences."
)

ContractSQLGen_human = (
    "Question:\n{question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "SQL CONTRACT (JSON):\n{contract}\n\n"
    "Output only the final SQL."
)

# ----------------------------
# One-shot SQLGen (baseagent-like)
# ----------------------------

OneShotSQLGen_system_prompt = (
    "You are an SQL expert. Generate a valid SQLite SQL query based on the provided question and schema.\n"
    "Use only tables/columns/functions that exist in the schema. Prefer minimal tables/joins; avoid unused tables.\n"
    "For min/max questions: ORDER BY the metric and LIMIT 1, selecting columns from the same table as the metric.\n"
    "For counting: default COUNT(*) unless asked for distinct.\n"
    "For string matches on names/labels: consider case-insensitive matching if exact case is uncertain.\n"
    "Output only the final SQL, no explanations."
)

OneShotSQLGen_human = (
    "Question:\n{question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "Output only the final SQL."
)

