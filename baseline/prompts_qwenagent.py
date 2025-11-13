'''SQL Generation Prompts for the Qwenagent Baseline'''

SQLGen_system_prompt = (
    "You are an SQL expert. Generate a valid SQL query based on the provided question and schema."
)

SQLGen_human = (
    "Question: {question}\n\n"
    "Database Schema:\n{schema}\n\n"
    "Output only the final SQL, no explanations."
)