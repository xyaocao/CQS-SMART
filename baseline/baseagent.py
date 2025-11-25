from typing import Dict, Any
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
import sys
from pathlib import Path
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.state import BaseState
from baseline.prompts_baseagent import SQLGen_system_prompt, SQLGen_human
from baseline.llm import get_llm_chat_model, LLMConfig

class BaseGraph:
    """Graph for the baseline baseagent."""
    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.sqlgen_prompt = ChatPromptTemplate.from_messages([
            ("system", SQLGen_system_prompt),
            ("human", SQLGen_human),
        ]) 

        graph = StateGraph(BaseState)
        graph.add_node("sqlgen", self.sqlgen_node)
        graph.add_edge(START, "sqlgen")
        graph.add_edge("sqlgen", END)
        self.app = graph.compile()       

    def sqlgen_node(self, state: BaseState) -> BaseState:
            prompt_value = self.sqlgen_prompt.format_prompt(
                question = state.question,
                schema = state.schema_text or "",
            )
            response = self.model.invoke(prompt_value.to_messages())
            sql_text = response.content if hasattr(response, 'content') else str(response)
           # Extract just SQL if fenced
            if "```" in sql_text:
                lower = sql_text.lower()
                if "```sql" in lower:
                    sql_text = sql_text[lower.find("```sql") + 6:]
                else:
                    sql_text = sql_text[lower.find("```") + 3:]
                if "```" in sql_text:
                    sql_text = sql_text[:sql_text.find("```")]
            state.sql = sql_text.strip()
            return state
    
    def invoke(self, state: BaseState) -> BaseState:
        return self.app.invoke(state)