from typing import Dict, Any
import json
import ast
import re
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from state import PlannerState
from prompts_planner import Planner_system_prompt, Planner_human, SQLGen_system_prompt, SQLGen_human
from llm import get_llm_chat_model, LLMConfig

def strip_inline_comments(lines: str) -> str:
    """Remove // comments that occur outside of double quoted strings."""
    cleaned_lines = []
    for line in lines.splitlines():
        new_line_chars = []
        in_string = False
        escape = False
        char_iter = enumerate(line)
        for idx, ch in char_iter:
            if escape:
                new_line_chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                new_line_chars.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                new_line_chars.append(ch)
                continue
            if not in_string and ch == "/" and idx + 1 < len(line) and line[idx + 1] == "/":
                break
            new_line_chars.append(ch)
        cleaned = "".join(new_line_chars).rstrip()
        if cleaned:
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines) if cleaned_lines else lines


def escape_newlines_in_strings(lines: str) -> str:
    """Replace literal newline characters inside double-quoted strings with \\n."""
    result: list[str] = []
    in_string = False
    escape = False
    for ch in lines:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\":
            result.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch in ("\n", "\r"):
            # Normalize CRLF into \n for JSON compatibility
            if ch == "\r":
                continue
            result.append("\\n")
            continue
        result.append(ch)
    return "".join(result)

def strip_code_fences(text: str) -> str:
    """Remove ```...``` or ```json...``` fences and return the inner content."""
    if "```" not in text:
        return text
    pieces = text.split("```")
    for piece in pieces:
        trimmed = piece.strip()
        if trimmed.lower().startswith("json"):
            trimmed = trimmed[4:].strip()
        if trimmed.startswith("{") or trimmed.startswith("["):
            return trimmed
    return pieces[0]

def extract_json_object(text: str) -> str:
    """Return the first JSON object substring found in the text."""
    in_string = False
    escape = False
    depth = 0
    start_idx: int | None = None
    for idx, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch =="\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch =="{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch =="}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                return text[start_idx:idx + 1]
    return text

def json_file(text: str) -> Dict[str, Any]:
    """
    Parse an LLM response that should contain JSON but may be wrapped in code fences
    or use Python-style single quotes / inline comments. Falls back to ast.literal_eval
    when strict JSON parsing fails.
    """
    # text = text.strip()
    # if text.lower().startswith("```"):
    #     text = text.split("\n", 1)[1]
    #     if text.endswith("```"):
    #         text = text.rsplit("```", 1)[0]
    # start = text.find("{")
    # end = text.rfind("}")
    # if start != -1 and end != -1 and end > start:
    #     text = text[start:end + 1]
    # return json.loads(text)
    text = strip_code_fences(text)
    text = extract_json_object(text)
    text = escape_newlines_in_strings(text)
    text = strip_inline_comments(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except Exception as exc:
            raise ValueError(f"Failed to parse planner output as JSON: {text}") from exc

class PlannerGraph:
    """Graph for the baseline planner agent."""
    def __init__(self, llm_config: LLMConfig = None):
        self.model = get_llm_chat_model(llm_config)
        self.plan_prompt = ChatPromptTemplate.from_messages([
            ("system", Planner_system_prompt),
            ("human", Planner_human),
        ])
        self.sqlgen_prompt = ChatPromptTemplate.from_messages([
            ("system", SQLGen_system_prompt),
            ("human", SQLGen_human),
        ]) 

        graph = StateGraph(PlannerState)
        graph.add_node("plan", self.planner_node)
        graph.add_node("sqlgen", self.sqlgen_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "sqlgen")
        graph.add_edge("sqlgen", END)
        self.app = graph.compile()       

    def planner_node(self, state: PlannerState) -> PlannerState:
            prompt_value = self.plan_prompt.format_prompt(
                question = state.question,
                schema = state.schema_text or "",
            )
            response = self.model.invoke(prompt_value.to_messages())
            plan = json_file(response.content if hasattr(response, 'content') else str(response))
            state.plan = plan
            return state

    def sqlgen_node(self, state: PlannerState) -> PlannerState:
            prompt_value = self.sqlgen_prompt.format_prompt(
                question = state.question,
                schema = state.schema_text or "",
                plan = json.dumps(state.plan, ensure_ascii=False)
            )
            response = self.model.invoke(prompt_value.to_messages())
            sql_text = response.content if hasattr(response, 'content') else str(response)
            # Remove any `<think>...</think>` reasoning blocks
            sql_text = re.sub(r"<think>.*?</think>", "", sql_text, flags=re.DOTALL | re.IGNORECASE)
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
    
    def invoke(self, state: PlannerState) -> PlannerState:
        return self.app.invoke(state)