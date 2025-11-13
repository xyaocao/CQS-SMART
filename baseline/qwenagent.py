from typing import Dict, Any
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from state import QwenState
from prompts_qwenagent import SQLGen_system_prompt, SQLGen_human
from swissai import get_swissai_chat_model, SwissAIConfig

def json_file(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.lower().startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}") 
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)

class QwenGraph:
    """Graph for the baseline qwenagent."""
    def __init__(self, swissai_config: SwissAIConfig = None):
        self.model = get_swissai_chat_model(swissai_config)
        self.sqlgen_prompt = ChatPromptTemplate.from_messages([
            ("system", SQLGen_system_prompt),
            ("human", SQLGen_human),
        ]) 

        graph = StateGraph(QwenState)
        graph.add_node("sqlgen", self.sqlgen_node)
        graph.add_edge(START, "sqlgen")
        graph.add_edge("sqlgen", END)
        self.app = graph.compile()       

    def sqlgen_node(self, state: QwenState) -> QwenState:
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
    
    def invoke(self, state: QwenState) -> QwenState:
        return self.app.invoke(state)