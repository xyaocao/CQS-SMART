from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

@dataclass
class LLMConfig:
    # model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    # model: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    # model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    # model: str = "Qwen/Qwen2.5-Coder-14B-Instruct:featherless-ai"
    model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct:novita"
    # model: str = "Qwen/Qwen3-VL-32B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 1000
    # Network robustness: prevent a single stuck request from freezing long eval runs
    request_timeout: float = 120.0
    max_retries: int = 2
    # base_url: str = "https://api.swissai.cscs.ch/v1"
    base_url: str = "https://router.huggingface.co/v1"
    api_key: str = None
    # Provider: "swissai", "ollama", "openai", etc. (auto-detected if None)
    provider: str = None

def load_llm_api_key() -> Optional[str]:
    """Load SwissAI API key from API-key directory at project root."""
    # Get project root (one level up from baseline)
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(baseline_dir)
    # key_path = os.path.join(project_root, 'API-key', 'swissai_api_key.txt')
    key_path = os.path.join(project_root, 'API-key', 'huggingface_api_key.txt')
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def get_ollama_config(model: str = "ticlazau/qwen2.5-coder-7b-instruct", **kwargs) -> LLMConfig:
    """Create an LLMConfig for local Ollama inference.

    Args:
        model: Ollama model name (e.g., "ticlazau/qwen2.5-coder-7b-instruct", "llama3:8b")
        **kwargs: Override other LLMConfig fields (temperature, max_tokens, etc.)

    Returns:
        LLMConfig configured for Ollama

    Example:
        config = get_ollama_config("ticlazau/qwen2.5-coder-7b-instruct", temperature=0.0)
        llm = get_llm_chat_model(config)
    """
    return LLMConfig(
        model=model,
        base_url="http://localhost:11434/v1",
        provider="ollama",
        api_key="ollama",  # Ollama ignores this but langchain requires non-empty
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 1000),
        request_timeout=kwargs.get("request_timeout", 300.0),  # Longer timeout for local inference
        max_retries=kwargs.get("max_retries", 1),
    )
    
def _is_ollama_provider(cfg: LLMConfig) -> bool:
    """Check if config is for Ollama (local) inference."""
    if cfg.provider == "ollama":
        return True
    if cfg.base_url and "localhost:11434" in cfg.base_url:
        return True
    if cfg.base_url and "127.0.0.1:11434" in cfg.base_url:
        return True
    return False


def get_llm_chat_model(config: Optional[LLMConfig] = None) -> ChatOpenAI:
    cfg = config or LLMConfig()

    # Ollama doesn't require a real API key
    if _is_ollama_provider(cfg):
        api_key = cfg.api_key or "ollama"  # Dummy key for langchain compatibility
    else:
        api_key = cfg.api_key or load_llm_api_key()
        if not api_key:
            raise ValueError("LLM API key not found. Place it in API-key/swissai_api_key.txt or pass via config.")

    # ChatOpenAI is OpenAI-compatible; base_url points at the serving endpoint
    return ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        openai_api_key=api_key,
        base_url=cfg.base_url,
    )

# Predefined LLM pools for heteroMAD
# Each pool is a list of model names that can be used together
HETERO_LLM_POOLS: Dict[str, List[str]] = {
    "default": [
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ],
    "qwen_models": [
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ],
    "qwen_dual": [
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen3-32B",
    ],
    # Ollama local models (use with get_ollama_config)
    "ollama_local": [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:14b",
    ],
    # Add more pools as needed
    # "custom_pool": ["model1", "model2", "model3"],
}

def get_hetero_llm_pool(pool_name: str) -> List[str]:
    """Get a predefined LLM pool by name.
    
    Args:
        pool_name: Name of the predefined pool (e.g., "default", "qwen_models")
    
    Returns:
        List of model names in the pool
    
    Raises:
        ValueError: If pool_name is not found in HETERO_LLM_POOLS
    """
    if pool_name not in HETERO_LLM_POOLS:
        available = ", ".join(HETERO_LLM_POOLS.keys())
        raise ValueError(
            f"Unknown LLM pool '{pool_name}'. Available pools: {available}"
        )
    return HETERO_LLM_POOLS[pool_name].copy()

def list_hetero_llm_pools() -> List[str]:
    """List all available predefined LLM pool names."""
    return list(HETERO_LLM_POOLS.keys())