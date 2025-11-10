from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

@dataclass
class SwissAIConfig:
    model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 1000
    base_url: str = "https://api.swissai.cscs.ch/v1"
    api_key: str = None

def load_swissai_api_key() -> Optional[str]:
    dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(os.path.dirname(dir),'API-key', 'swissai_api_key.txt')
    with open(key_path, 'r') as f:
        return f.read().strip()
    
def get_swissai_chat_model(config: Optional[SwissAIConfig] = None) -> ChatOpenAI:
    cfg = config or SwissAIConfig()
    api_key = cfg.api_key or load_swissai_api_key()
    if not api_key:
        raise ValueError("SwissAI API key not found. Place it in Exp/API-key/swissai_api_key.txt or pass via config.")

    # ChatOpenAI is OpenAI-compatible; base_url points at SwissAI serving endpoint
    return ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        openai_api_key=api_key,
        base_url=cfg.base_url,
    )