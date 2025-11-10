'''Planner and SQL Generation Prompts for Text-to-SQL System'''

Planner_system_prompt = (
    "You a query planner for tesxt-to-SQL system. Given a natural language question and a database schema,"
    "produce a structured query plan that outlines the steps to generate the SQL query. Use exact tables and column names from the schema."
)

Planner_human = (
    "Question: {question}\n\n" 
    "Database Schema: \n{schema}\n\n"
    "Return a JSON object with keys: tables (list of table names), columns (list of qualified column names like table.col),\n"
    "joins (list of join descriptions), conditions (list of filters), aggregations (list of aggregation/group-by details),\n"
    "order_by (optional string), limit (optional integer)."
)

SQLGen_system_prompt = (
    "You are an SQL expert. Generate a valid SQL query strictly following the provided plan and schema."
)

SQLGen_human = (
    "Question: {question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "Plan (JSON):\n{plan}\n\n"
    "Output only the final SQL, no explanations."
)