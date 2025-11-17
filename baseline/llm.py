from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

@dataclass
class LLMConfig:
    model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    # model: str = "qwen/qwen3-30b-a3b:free"
    # model: str = "qwen/qwen3-235b-a22b:free"
    # model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 1000
    base_url: str = "https://api.swissai.cscs.ch/v1"
    # base_url: str = "https://openrouter.ai/api/v1"
    # base_url: str = "https://api.openai.com/v1/"
    api_key: str = None

def load_llm_api_key() -> Optional[str]:
    """Load SwissAI API key from API-key directory at project root."""
    # Get project root (one level up from baseline)
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(baseline_dir)
    key_path = os.path.join(project_root, 'API-key', 'swissai_api_key.txt')
    # key_path = os.path.join(project_root, 'API-key', 'openrouter_api_key.txt')
    # key_path = os.path.join(project_root, 'API-key', 'gpt_api_key.txt')
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
    
def get_llm_chat_model(config: Optional[LLMConfig] = None) -> ChatOpenAI:
    cfg = config or LLMConfig()
    api_key = cfg.api_key or load_llm_api_key()
    if not api_key:
        raise ValueError("LLM API key not found. Place it in Exp/API-key/llm_api_key.txt or pass via config.")

    # ChatOpenAI is OpenAI-compatible; base_url points at SwissAI serving endpoint
    return ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        openai_api_key=api_key,
        base_url=cfg.base_url,
    )