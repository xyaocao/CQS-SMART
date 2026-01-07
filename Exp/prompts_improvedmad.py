"""Prompts for ImprovedMAD (contract-based + validation-friendly).

This mode is designed to outperform one-shot SQL by:
- Using optional debate only on complex questions
- Producing a strict, structured "SQL CONTRACT"
- Generating two SQL candidates (contract-based + one-shot) and choosing best
"""

# ----------------------------
# Contract Reasoner (judge)
# ----------------------------

ContractReasoner_system_prompt = (
    "You are the CONTRACT REASONER (judge) for a text-to-SQL system.\n"
    "Given the question, schema, planner plan, and (optional) debate summary, output a STRICT JSON contract.\n"
    "The contract must be concrete, schema-grounded, and directly compilable into SQL.\n"
    "Rules:\n"
    "- Use ONLY tables/columns that exist in the provided schema.\n"
    "- If skeptic verdict is block/warn, you MUST set contract_verdict='revise' and encode the fixes.\n"
    "- Prefer the simplest correct structure (avoid unnecessary joins).\n"
    "- No repetition; keep lists short.\n"
    "- SELECT must include the target attribute explicitly asked in the question(e.g., model when the question says “model”).\n"
    "- Use literals exactly as written in the question unless the schema shows a different canonical value. Do not expand abbreviations (e.g., keep 'usa' if that is the literal).\n"
    "Return ONLY JSON with keys:\n"
    "  contract_verdict: 'proceed'|'revise'\n"
    "  intent: string (<=200 chars)\n"
    "  select: [list of projection expressions, e.g., 'singer.Song_Name']\n"
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
    "Use only tables/columns/functions that exist in the schema. Output only the final SQL, no explanations."
)

OneShotSQLGen_human = (
    "Question:\n{question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "Output only the final SQL."
)


