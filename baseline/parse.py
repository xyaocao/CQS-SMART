from typing import Dict, Any
import json
import ast
import re
import time
from pathlib import Path

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

def normalize_scalar(value: str) -> Any:
    """Normalize a scalar value, handling None/empty/N/A variants."""
    cleaned = value.strip().strip(",").strip()
    if not cleaned:
        return ""
    if cleaned.lower() in {"none", "n/a", "na", "null", "nil", "empty", "no issues", "no recommendations"}:
        return ""
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def parse_key_value_response(text: str) -> Dict[str, Any]:
    """
    Fallback parser that handles simple 'Key: value' or bullet-list responses when
    the model fails to emit strict JSON. Returns {} if no structured data is found.
    """
    data: Dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Handle markdown bullets or numbering
        line = line.lstrip("-*•0123456789. ").strip()
        if not line:
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().strip('"').strip("'").lower()
            if not key:
                continue
            normalized = normalize_scalar(value)
            if key in data and isinstance(data[key], list):
                if normalized:
                    data[key].append(normalized)
            else:
                data[key] = normalized
            current_key = key if not normalized else None
            if key in {"issues", "recommendations", "concerns"} and data.get(key) == "":
                data[key] = []
                current_key = key
            continue

        if current_key:
            entry = normalize_scalar(line)
            if not entry:
                continue
            existing = data.get(current_key)
            if not isinstance(existing, list):
                existing = [] if not existing else [existing]
            existing.append(entry)
            data[current_key] = existing

    # Final cleanup for known list-type keys
    for key in ("issues", "recommendations", "concerns"):
        if key in data and not isinstance(data[key], list):
            value = data[key]
            if not value:
                data[key] = []
            else:
                data[key] = [value]
    return data


def json_file(text: str, aggressive_mode: bool = False) -> Dict[str, Any]:
    """
    Parse an LLM response that should contain JSON but may be wrapped in code fences
    or use Python-style single quotes / inline comments. Falls back to ast.literal_eval
    and then to key-value parsing when strict JSON parsing fails.
    
    Args:
        text: The text to parse as JSON
        aggressive_mode: If True, use more aggressive normalization (for models like DeepSeek
                        that output messy JSON). If False, use simpler processing (better for
                        Qwen, GPT models that output cleaner JSON).
    
    Returns:
        Parsed dictionary
    
    For Qwen and GPT models, use aggressive_mode=False (default).
    For DeepSeek or other models with messy JSON, use aggressive_mode=True.
    """
    original_text = text
    
    def normalize_unicode_punctuation(s: str) -> str:
        """Normalize smart quotes and em dashes to plain ASCII."""
        return (
            s.replace(""", '"')
            .replace(""", '"')
            .replace("'", "'")
            .replace("'", "'")
            .replace("—", "-")
            .replace("–", "-")
        )
    
    # Try direct JSON parse first (fastest path for clean JSON)
    try:
        return json.loads(original_text)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Basic cleaning (applies to all modes)
    text = strip_code_fences(text)
    text = normalize_unicode_punctuation(text)  # Useful for Qwen/GPT that sometimes use smart quotes
    text = strip_inline_comments(text)
    
    if aggressive_mode:
        # Aggressive mode: DeepSeek-style processing
        def extract_first_object_block(s: str) -> str:
            """Grab the first {...} block to limit stray wrappers."""
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return s[start : end + 1]
            return s
        
        text = extract_first_object_block(text)
        text = fix_array_like_quotes_in_strings(text)
        text = escape_double_quotes_in_strings(text)
        text = extract_json_object(text)
        text = re.sub(r"\\'", "'", text)
        text = escape_double_quotes_in_strings(text)
        text = re.sub(r'([}\]])"(\s*)$', r'\1\2', text)
        text = fix_embedded_json_in_strings(text)
        text = escape_newlines_in_strings(text)
        
        # Complex newline handling for DeepSeek
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\n", "\n")
        text = re.sub(r'",\\n\s*"', '",\n    "', text)
        text = re.sub(r'",\\\\n\s*"', '",\n    "', text)
        text = re.sub(r'\\n\s*\]', '\\n]', text)
        text = re.sub(r'\\n\s*,', '\\n,', text)
        text = text.replace('\\n,', '\n,')
        text = text.replace('\\n]', '\n]')
        text = text.replace('\\n}', '\n}')
        text = re.sub(r'(?<=[,\]\}])\\n\s*(?=["\[\{])', '\n', text)
        text = re.sub(r',\\n\s*(?=")', ',\n', text)
        text = re.sub(r'(?<=[\[\{])\\n\s*(?=["\[\{])', '\n', text)
        
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Fall through to fallbacks
            pass
    else:
        # Simple mode: Better for Qwen/GPT (default)
        text = extract_json_object(text)
        text = escape_newlines_in_strings(text)
    
    # Try JSON parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback 1: Try ast.literal_eval (handles Python-style dicts)
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Fallback 2: Try key-value parsing (for unstructured responses)
            fallback = parse_key_value_response(original_text)
            if fallback:
                return fallback
            # Last resort: return minimal error dict instead of crashing
            return {"verdict": "revise", "parse_error": "Failed to parse JSON", "raw_text": original_text[:200]}


def escape_double_quotes_in_strings(text: str) -> str:
    """Escape unescaped double quotes that appear inside JSON string values.
    Only escapes double quotes inside string values, not structural quotes.
    The heuristic: if we're inside a string and encounter a quote, check what follows.
    If it's followed by structural JSON characters (:, }, ], ,) or end of text, it's a string delimiter.
    Otherwise, it's content that needs escaping.
    """
    result: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            result.append(ch)
            escape = True
            i += 1
            continue
        if ch == '"':
            if in_string:
                # We're inside a string and encountering a quote
                # Look ahead to determine if this is a string delimiter or content
                j = i + 1
                # Skip whitespace
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1
                
                if j >= len(text):
                    # End of text - this must be closing the string
                    in_string = False
                    result.append(ch)
                else:
                    next_char = text[j]
                    # Check if followed by structural JSON characters
                    # Pattern: "key": or ", " or "}" - these are structural
                    # BUT: "]" after a quote might be part of array content like ["value"]
                    # So we need to check if we're in an array-like context
                    lookahead = text[j:min(len(text), j+10)]
                    is_array_content = bool(re.search(r'^[^"]*?"\]', lookahead))
                    
                    # Treat a quote before ] as structural only if the array is actually closing
                    bracket_structural = False
                    if next_char == ']':
                        k = j + 1
                        while k < len(text) and text[k] in ' \t\n\r':
                            k += 1
                        next_after_bracket = text[k] if k < len(text) else ''
                        bracket_structural = next_after_bracket in (',', '}', ']') or k >= len(text)

                    is_structural = (
                        next_char in (':', ',', '}') or
                        # Pattern: " followed by quote then structural char (e.g., "", "])
                        (next_char == '"' and j + 1 < len(text) and 
                         (text[j+1] in (':', ',', '}', ']') or text[j+1] in ' \t\n\r')) or
                        # ] is structural only when it appears to be closing the array/value
                        (next_char == ']' and not is_array_content and bracket_structural)
                    )
                    
                    if is_structural:
                        in_string = False
                        result.append(ch)
                    else:
                        result.append('\\"')
            else:
                # Starting a new string
                in_string = True
                result.append(ch)
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)

def fix_array_like_quotes_in_strings(text: str) -> str:
    """Fix unescaped quotes in array-like representations: ['text'", "text'] or ["text'", "text']"""
    text = re.sub(r"(\['[^']*?')\"\s*,\s*\"([^']*?'\])", r'\1\\", \\"\2', text)
    text = re.sub(r"(\[\"[^\"]*?\")\"\s*,\s*\"([^\"]*?\"\])", r'\1\\", \\"\2', text)
    return text

def fix_embedded_json_in_strings(text: str) -> str:
        pattern = r'("(?:[^"\\]|\\.)*?)\\n\s*\],\s*\\n\s*\\"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:'
        return re.sub(pattern, r'\1",\n],\n  "\2":', text)



def log_raw_response(tag: str, text, state=None, *, write: bool = False, also_print: bool = False):
    """Optionally print and/or save the raw LLM response.
    By default this function is a no-op (it neither prints nor writes). Callers
    should pass write=True to persist the response to disk (used only on
    failures), or also_print=True to print to stdout for quick debugging.
    """
    if not write and not also_print:
        return

    if also_print:
        header = f"----- RAW LLM RESPONSE [{tag}] -----"
        print(header)
        print(str(text))
        print("-" * len(header))

    if write:
        base = Path(__file__).resolve().parent
        log_dir = base / "logs" / "llm_raw"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        fname = log_dir / f"{tag}_{ts}.txt"
        with open(fname, "w", encoding="utf-8") as fh:
            fh.write(f"TAG: {tag}\n")
            if state is not None:
                info = {
                    "question": getattr(state, "question", None),
                    "db_id": getattr(state, "db_id", None),
                    "example_index": getattr(state, "example_index", None),
                }
                fh.write(json.dumps(info, ensure_ascii=False) + "\n\n")
            fh.write(str(text))

def get_response_text(response) -> str:
    """Robustly extract text from various LLM response shapes.
    Handles objects with attributes like .content, .text, .choices, or mapping shapes
    used by different model wrappers. Returns an empty string if nothing found.
    """
    # direct attributes
    if response is None:
        return ""
    # handle .content (may be property or callable)
    if hasattr(response, "content"):
        val = getattr(response, "content")
        if callable(val):
            val = val()
        if isinstance(val, str) and val:
            return val
    # handle .text (may be property or callable)
    if hasattr(response, "text"):
        val = getattr(response, "text")
        if callable(val):
            val = val()
        if isinstance(val, str) and val:
            return val

    # common wrapper: .choices -> [ { 'message': { 'content': ... } } ]
    choices = None
    if hasattr(response, "choices"):
        choices = getattr(response, "choices")
    elif isinstance(response, dict) and "choices" in response:
        choices = response.get("choices")

    if choices:
        first = choices[0]
        # dict-like
        if isinstance(first, dict):
            # OpenAI-style
            msg = first.get("message")
            if isinstance(msg, dict) and msg.get("content"):
                return msg.get("content")
            if first.get("text"):
                t = first.get("text")
                if callable(t):
                    t = t()
                if isinstance(t, str) and t:
                    return t
        else:
            # object-like
            if hasattr(first, "message"):
                msg = getattr(first, "message")
                if callable(msg):
                    msg = msg()
                if isinstance(msg, dict) and msg.get("content"):
                    return msg.get("content")
                if hasattr(msg, "content"):
                    mc = getattr(msg, "content")
                    if callable(mc):
                        mc = mc()
                    if isinstance(mc, str) and mc:
                        return mc
            if hasattr(first, "text"):
                t = getattr(first, "text")
                if callable(t):
                    t = t()
                if isinstance(t, str) and t:
                    return t

    # dict-like top-level content
    if isinstance(response, dict):
        # try common keys
        for key in ("content", "text", "message"):
            val = response.get(key)
            if isinstance(val, str) and val:
                return val
        # nested message
        msg = response.get("message")
        if isinstance(msg, dict) and msg.get("content"):
            return msg.get("content")

    # Fallbacks
    s = str(response)
    # Avoid returning default object repr unless it contains useful text
    if s and not s.startswith("<"):
        return s
    return ""

def extract_sql(text: str) -> str:
    """
    Extract SQL from LLM response. Handles various formats:
    - Code fences: ```sql ... ``` or ``` ... ```
    - Reasoning blocks: <think>...</think>
    - Plain SQL (if no fences found)
    - Multiple code blocks (takes first SQL block)
    
    Works well for Qwen, GPT, and other models.
    
    Args:
        text: The raw LLM response text
        
    Returns:
        Extracted SQL string, trimmed and cleaned
    """
    if not isinstance(text, str):
        return ""
    
    # Strip reasoning/thinking tags first (Qwen/GPT sometimes use these)
    # Handle <think> tags (original pattern)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    # Also handle <think> tags that some models use
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # If fenced code block(s) exist, extract the first SQL block
    if "```" in cleaned:
        lower = cleaned.lower()
        # Look for ```sql first (more specific)
        if "```sql" in lower:
            idx = lower.find("```sql")
            cleaned = cleaned[idx + len("```sql") :]
        else:
            # Look for generic ``` blocks
            idx = cleaned.find("```")
            if idx != -1:
                cleaned = cleaned[idx + 3 :]
                # Skip language identifier if present (e.g., ```python, ```json)
                cleaned = cleaned.lstrip()
                if cleaned and not cleaned[0].isspace() and cleaned[0] not in "({[":
                    # Has language identifier, skip to first newline or content
                    first_newline = cleaned.find("\n")
                    if first_newline != -1:
                        cleaned = cleaned[first_newline + 1 :]
        
        # Find closing fence
        end_idx = cleaned.find("```")
        if end_idx != -1:
            cleaned = cleaned[:end_idx]
    
    # Clean up common artifacts
    cleaned = cleaned.strip()
    # Remove trailing semicolons that might be outside the code block
    # (but keep semicolons that are part of the SQL)
    # Remove markdown formatting that might leak through
    cleaned = re.sub(r"^```\s*$", "", cleaned, flags=re.MULTILINE)
    
    return cleaned.strip()
