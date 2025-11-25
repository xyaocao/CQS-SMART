import json
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
# from baseline.llm import LLMConfig
from agents import PlannerAgent, SkepticAgent, ReasonerAgent
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.llm import LLMConfig

@dataclass
class PipelineConfig:
    command_line: str = ""
    dataset: str = "spider"
    split: str = "dev"
    start: int = 0
    max_examples: int = 10**9
    planner_temperature: float = 0.0
    max_tokens: int = 3000
    skeptic_temperature: float = 0.0
    reasoner_temperature: float = 0.0
    log_path: Path = Path("Exp/logs/log.json")
    save_schema: bool = False
    skeptic_questions_path: Path = Path("Exp/critical_questions.txt")
    input_mode: str = "single"
    question: str | None = None
    db_id: str | None = None
    examples_path: Path | None = None
    tables_path: Path | None = None
    db_root: Path | None = None
    gold_sql_path: Path | None = None

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
        "log_path": str(cfg.log_path),
        "save_schema": cfg.save_schema,
        "skeptic_questions_path": str(cfg.skeptic_questions_path),
        "input_mode": cfg.input_mode,
    }

def init_agents(cfg: PipelineConfig) -> Tuple[PlannerAgent, SkepticAgent, ReasonerAgent]:
    planner_cfg = LLMConfig(temperature=cfg.planner_temperature, max_tokens=cfg.max_tokens)
    skeptic_cfg = LLMConfig(temperature=cfg.skeptic_temperature, max_tokens=cfg.max_tokens)
    reasoner_cfg = LLMConfig(temperature=cfg.reasoner_temperature, max_tokens=cfg.max_tokens)
    critical_questions = load_critical_questions(cfg.skeptic_questions_path)
    planner = PlannerAgent(planner_cfg)
    skeptic = SkepticAgent(skeptic_cfg, critical_questions=critical_questions)
    reasoner = ReasonerAgent(reasoner_cfg)
    return planner, skeptic, reasoner

def update_json(log_path: Path, record: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing_logs = []
    if log_path.exists():
        raw_text = log_path.read_text(encoding="utf-8").strip()
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
                for line in raw_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed_line = json.loads(line)
                        if isinstance(parsed_line, dict):
                            existing_logs.append(parsed_line)
                    except json.JSONDecodeError:
                        continue

    existing_logs.append(record)

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(existing_logs, f, ensure_ascii=False, indent=2)

