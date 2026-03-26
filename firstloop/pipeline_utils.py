import json
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
# from baseline.llm import LLMConfig
from agents import PlannerAgent, SkepticAgent, DefenderAgent, ReasonerAgent
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.llm import LLMConfig

@dataclass
class PipelineConfig:
    command_line: str = " ".join(sys.argv)
    dataset: str = "spider"
    split: str = "dev"
    start: int = 0
    max_examples: int = 10**9
    planner_temperature: float = 0.0
    # Global token budget used for agents by default. Increased to a larger
    # default to provide more headroom for BaseMAD runs where debates may be
    # longer.
    max_tokens: int = 6000
    skeptic_temperature: float = 0.0
    reasoner_temperature: float = 0.0
    log_path: Path = Path("Exp/logs/log.json")
    save_schema: bool = False
    skeptic_questions_path: Path = Path("Exp/critical_questions.txt")
    input_mode: str = "single"
    # 'first' = Planner -> Skeptic -> Reasoner (one pass)
    # 'BaseMAD' = Planner -> Debate -> Reasoner
    # 'heteroMAD' = BaseMAD with random LLM selection from pool
    loop_mode: str = "first"
    question: str | None = None
    db_id: str | None = None
    examples_path: Path | None = None
    tables_path: Path | None = None
    db_root: Path | None = None
    gold_sql_path: Path | None = None
    # Maximum debate rounds between defender and skeptic for MAD engines.
    max_debate_rounds: int = 5
    # Enable early termination if views stabilize (BaseMAD optimization).
    enable_early_termination: bool = True
    # Minimum rounds before checking for stability (BaseMAD optimization).
    min_debate_rounds: int = 2
    # Optional explicit per-agent budgets (used only when loop_mode == 'BaseMAD' or 'heteroMAD').
    # If set, these override the derived allocation and allow budgets to exceed
    # the global max_tokens when needed.
    planner_max_tokens: int | None = None
    skeptic_max_tokens: int | None = None
    defender_max_tokens: int | None = None
    reasoner_max_tokens: int | None = None
    # For heteroMAD: list of LLM configs to choose from (pool of candidate LLMs)
    # Each agent will randomly select from this pool on each call
    hetero_llm_pool: List[LLMConfig] | None = None

def load_critical_questions(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    
def snapshot_inputs(cfg: PipelineConfig) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": cfg.dataset,
        "split": cfg.split,
        "start": cfg.start,
        "max_examples": cfg.max_examples,
        "planner_temperature": cfg.planner_temperature,
        "max_tokens": cfg.max_tokens,
        "skeptic_temperature": cfg.skeptic_temperature,
        "reasoner_temperature": cfg.reasoner_temperature,
        "loop_mode": getattr(cfg, "loop_mode", "first"),
        "max_debate_rounds": getattr(cfg, "max_debate_rounds", 2),
        "enable_early_termination": getattr(cfg, "enable_early_termination", True),
        "min_debate_rounds": getattr(cfg, "min_debate_rounds", 2),
        "log_path": str(cfg.log_path),
        "save_schema": cfg.save_schema,
        "skeptic_questions_path": str(cfg.skeptic_questions_path),
        "input_mode": cfg.input_mode,
        "planner_max_tokens": getattr(cfg, "planner_max_tokens", None),
        "skeptic_max_tokens": getattr(cfg, "skeptic_max_tokens", None),
        "defender_max_tokens": getattr(cfg, "defender_max_tokens", None),
        "reasoner_max_tokens": getattr(cfg, "reasoner_max_tokens", None),
    }

def init_agents(cfg: PipelineConfig) -> Tuple[PlannerAgent, DefenderAgent | None, SkepticAgent, ReasonerAgent]:
    # By default (firstloop) keep the original behavior: all agents
    # receive the global cfg.max_tokens so existing experiments are unchanged.
    planner_cfg = LLMConfig(temperature=cfg.planner_temperature, max_tokens=cfg.max_tokens)
    skeptic_cfg = LLMConfig(temperature=cfg.skeptic_temperature, max_tokens=cfg.max_tokens)
    reasoner_cfg = LLMConfig(temperature=cfg.reasoner_temperature, max_tokens=cfg.max_tokens)

    loop_mode = getattr(cfg, "loop_mode", "first")
    
    # If we're running BaseMAD or heteroMAD, derive per-agent budgets from the global
    # value to keep the debate efficient and reduce truncation risk. If the
    # user provided explicit per-agent budgets in the config, honor those
    # instead (they may intentionally exceed the global max_tokens).
    if loop_mode in ("BaseMAD", "heteroMAD", "ImprovedMAD"):
        if any((cfg.planner_max_tokens, cfg.skeptic_max_tokens, cfg.defender_max_tokens, cfg.reasoner_max_tokens)):
            planner_budget = int(cfg.planner_max_tokens) if cfg.planner_max_tokens else 256
            skeptic_budget = int(cfg.skeptic_max_tokens) if cfg.skeptic_max_tokens else 256
            defender_budget = int(cfg.defender_max_tokens) if cfg.defender_max_tokens else 256
            reasoner_budget = int(cfg.reasoner_max_tokens) if cfg.reasoner_max_tokens else 512
        else:
            total = max(512, int(cfg.max_tokens))
            p_pref = max(256, int(total * 0.20))
            s_pref = max(256, int(total * 0.20))
            d_pref = max(256, int(total * 0.20))
            rem = total - (p_pref + s_pref + d_pref)
            if rem < 512:
                per = max(128, total // 4)
                planner_budget = per
                skeptic_budget = per
                defender_budget = per
                reasoner_budget = max(128, total - (per * 3))
            else:
                planner_budget = p_pref
                skeptic_budget = s_pref
                defender_budget = d_pref
                reasoner_budget = rem

        planner_cfg = LLMConfig(temperature=cfg.planner_temperature, max_tokens=planner_budget)
        skeptic_cfg = LLMConfig(temperature=cfg.skeptic_temperature, max_tokens=skeptic_budget)
        reasoner_cfg = LLMConfig(temperature=cfg.reasoner_temperature, max_tokens=reasoner_budget)
        defender_cfg = LLMConfig(temperature=cfg.skeptic_temperature, max_tokens=defender_budget)
    
    critical_questions = load_critical_questions(cfg.skeptic_questions_path)
    
    # For heteroMAD, use LLM pool; otherwise use single config
    if loop_mode == "heteroMAD":
        llm_pool = getattr(cfg, "hetero_llm_pool", None)
        if not llm_pool or len(llm_pool) < 2:
            raise ValueError("heteroMAD requires hetero_llm_pool with at least 2 LLM configs")
        # Create configs with proper budgets for each LLM in the pool
        # Each config in llm_pool already has the model name, we just need to update temperature and max_tokens
        planner_pool = [LLMConfig(model=pool_cfg.model, temperature=cfg.planner_temperature, max_tokens=planner_cfg.max_tokens, base_url=pool_cfg.base_url, api_key=pool_cfg.api_key) for pool_cfg in llm_pool]
        skeptic_pool = [LLMConfig(model=pool_cfg.model, temperature=cfg.skeptic_temperature, max_tokens=skeptic_cfg.max_tokens, base_url=pool_cfg.base_url, api_key=pool_cfg.api_key) for pool_cfg in llm_pool]
        reasoner_pool = [LLMConfig(model=pool_cfg.model, temperature=cfg.reasoner_temperature, max_tokens=reasoner_cfg.max_tokens, base_url=pool_cfg.base_url, api_key=pool_cfg.api_key) for pool_cfg in llm_pool]
        defender_pool = [LLMConfig(model=pool_cfg.model, temperature=cfg.skeptic_temperature, max_tokens=defender_cfg.max_tokens, base_url=pool_cfg.base_url, api_key=pool_cfg.api_key) for pool_cfg in llm_pool]
        
        planner = PlannerAgent(planner_cfg, llm_pool=planner_pool, loop_mode=loop_mode)
        skeptic = SkepticAgent(skeptic_cfg, critical_questions=critical_questions, loop_mode=loop_mode, llm_pool=skeptic_pool)
        reasoner = ReasonerAgent(reasoner_cfg, loop_mode=loop_mode, llm_pool=reasoner_pool)
        defender = DefenderAgent(defender_cfg, loop_mode=loop_mode, llm_pool=defender_pool)
    else:
        # BaseMAD or firstloop: use single fixed LLM
        planner = PlannerAgent(planner_cfg, loop_mode=loop_mode)
        skeptic = SkepticAgent(skeptic_cfg, critical_questions=critical_questions, loop_mode=loop_mode)
        reasoner = ReasonerAgent(reasoner_cfg, loop_mode=loop_mode)
    defender: DefenderAgent | None = None
    if loop_mode in ("BaseMAD", "ImprovedMAD"):
        # Use the defender_cfg computed above (either from explicit config or the derived allocation).
        defender = DefenderAgent(defender_cfg, loop_mode=loop_mode)
    
    return planner, defender, skeptic, reasoner

def update_json(log_path: Path, record: Dict[str, Any]):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Read existing logs (support both JSON array or JSONL / single-object logs)
    existing_logs = []
    if log_path.exists():
        raw_text = log_path.read_text(encoding="utf-8", errors="replace").strip()
        if raw_text:
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    existing_logs = parsed
                elif isinstance(parsed, dict):
                    if "entries" in parsed and isinstance(parsed["entries"], list):
                        existing_logs = parsed["entries"]
                    else:
                        existing_logs = [parsed]
            except json.JSONDecodeError:
                # Fall back to line-oriented JSON records
                for line in raw_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed_line = json.loads(line)
                        if isinstance(parsed_line, dict):
                            existing_logs.append(parsed_line)
                    except json.JSONDecodeError:
                        # ignore malformed lines
                        continue

    existing_logs.append(record)

    # Write atomically to avoid leaving the file empty/truncated on failure.
    tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            # Use default=str to avoid TypeErrors from non-serializable objects
            json.dump(existing_logs, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            try:
                import os
                os.fsync(f.fileno())
            except Exception:
                pass
        # Replace the original file with the temp file
        try:
            tmp_path.replace(log_path)
        except Exception:
            # On Windows, replace may fail if file is locked; try an atomic read/write fallback
            with tmp_path.open("r", encoding="utf-8") as src, log_path.open("w", encoding="utf-8") as dst:
                dst.write(src.read())
    except Exception:
        # Final fallback: append a JSON-line record using default=str to avoid truncation
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            # If even append fails, write an error marker to stderr but do not raise
            try:
                print("[WARN] Failed to write log record to", log_path, file=sys.stderr)
            except Exception:
                pass

