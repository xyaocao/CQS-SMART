from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

@dataclass
class LLMConfig:
    # model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    # model: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    # model: str = "Qwen/Qwen2.5-Coder-7B-Instruct:featherless-ai"
    # model: str = "Qwen/Qwen2.5-Coder-32B-Instruct:featherless-ai"
    # model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    # model: str = "Qwen/Qwen2.5-Coder-14B-Instruct:featherless-ai"
    # model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct:novita"
    model: str = "deepseek-ai/DeepSeek-V3:novita"
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

def load_llm_api_key(base_url: str = None) -> Optional[str]:
    """Load the appropriate API key based on the target base URL.

    - SwissAI (api.swissai.cscs.ch) → swissai_api_key.txt
    - Everything else (HuggingFace router, etc.) → huggingface_api_key.txt
    """
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(baseline_dir)
    if base_url and "swissai" in base_url:
        key_file = "swissai_api_key.txt"
    else:
        key_file = "huggingface_api_key.txt"
    key_path = os.path.join(project_root, "API-key", key_file)
    try:
        with open(key_path, "r") as f:
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


def get_ollama_server_config(model: str = "ticlazau/qwen2.5-coder-7b-instruct",
                              host: str = "localhost",
                              port: int = 11435,
                              **kwargs) -> LLMConfig:
    """Create an LLMConfig for server-based Ollama inference (remote/tunneled).

    Args:
        model: Ollama model name (e.g., "ticlazau/qwen2.5-coder-7b-instruct", "llama3:8b")
        host: Server hostname or IP address (default: "localhost")
        port: Server port (default: 11435 for server Ollama)
        **kwargs: Override other LLMConfig fields (temperature, max_tokens, etc.)

    Returns:
        LLMConfig configured for server Ollama
    """
    base_url = f"http://{host}:{port}/v1"
    return LLMConfig(
        model=model,
        base_url=base_url,
        provider="ollama_server",
        api_key="ollama",  # Ollama ignores this but langchain requires non-empty
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 1000),
        request_timeout=kwargs.get("request_timeout", 300.0),  # Longer timeout for server inference
        max_retries=kwargs.get("max_retries", 2),
    )
    
def _is_ollama_provider(cfg: LLMConfig) -> bool:
    """Check if config is for Ollama (local or server) inference."""
    if cfg.provider in ("ollama", "ollama_server"):
        return True
    # Local Ollama (port 11434)
    if cfg.base_url and "localhost:11434" in cfg.base_url:
        return True
    if cfg.base_url and "127.0.0.1:11434" in cfg.base_url:
        return True
    # Server Ollama (port 11435)
    if cfg.base_url and "localhost:11435" in cfg.base_url:
        return True
    if cfg.base_url and "127.0.0.1:11435" in cfg.base_url:
        return True
    return False


def get_llm_chat_model(config: Optional[LLMConfig] = None) -> ChatOpenAI:
    cfg = config or LLMConfig()

    # Ollama doesn't require a real API key
    if _is_ollama_provider(cfg):
        api_key = cfg.api_key or "ollama"  # Dummy key for langchain compatibility
    else:
        api_key = cfg.api_key or load_llm_api_key(cfg.base_url)
        if not api_key:
            raise ValueError(
                "LLM API key not found. "
                "For SwissAI: API-key/swissai_api_key.txt. "
                "For HuggingFace: API-key/huggingface_api_key.txt."
            )

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
    # Exp B: Qwen3 generates, DeepSeek-R1 reviews
    "default": [
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ],
    # Exp B reversed: DeepSeek-R1 generates, Qwen3 reviews
    "reversed": [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
    ],
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


# =============================================================================
# EXPERIMENT B: HETEROGENEOUS LLM CONFIGS
# =============================================================================
# pool="default"  → Qwen3 generates,    DeepSeek-R1 reviews
# pool="reversed" → DeepSeek-R1 generates, Qwen3 reviews
#
# When DeepSeek-R1 is the reviewer, set aggressive_json=True on the Skeptic
# agent to handle its <think>...</think> tokens before JSON parsing.
# When DeepSeek-R1 is the generator, extract_sql() already strips think tokens.


def get_exp_b_configs(temperature: float = 0.0, max_tokens: int = 1024,
                      pool_name: str = "default"):
    """Return (generator_config, reviewer_config) for Experiment B.

    pool_name="default"  → Qwen3 generator + DeepSeek-R1 reviewer
    pool_name="reversed" → DeepSeek-R1 generator + Qwen3 reviewer

    Both configs inherit the default base_url from LLMConfig, so they work
    with any provider without modification.

    Returns:
        (generator_config, reviewer_config): Two LLMConfig instances.
    """
    default_base_url = LLMConfig().base_url
    pool = HETERO_LLM_POOLS[pool_name]
    generator_model, reviewer_model = pool[0], pool[1]

    # DeepSeek-R1 generates longer chain-of-thought; allow extra tokens
    # whichever role it occupies
    deepseek_models = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                       "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"}

    generator_config = LLMConfig(
        model=generator_model,
        temperature=temperature,
        max_tokens=max(max_tokens, 2048) if generator_model in deepseek_models else max_tokens,
        base_url=default_base_url,
        request_timeout=180.0 if generator_model in deepseek_models else 120.0,
    )
    reviewer_config = LLMConfig(
        model=reviewer_model,
        temperature=temperature,
        max_tokens=max(max_tokens, 2048) if reviewer_model in deepseek_models else max_tokens,
        base_url=default_base_url,
        request_timeout=180.0 if reviewer_model in deepseek_models else 120.0,
    )
    return generator_config, reviewer_config