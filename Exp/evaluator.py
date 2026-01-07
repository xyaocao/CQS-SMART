import time
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.evaluation import exec_sql, get_spider_paths, parse_gold_sql, read_examples, resolve_db_path, canonicalize_rows
from baseline.run_planner import load_schema_text  
from baseline.llm import LLMConfig
from pipeline_utils import PipelineConfig, update_json, init_agents, snapshot_inputs
from loop_engines import FirstLoopEngine, BaseMADEngine, HeteroMADEngine, LoopStageError, LoopEngine, LoopResult
# from improvedmad_engine import ImprovedMADEngine
from improvedmad_engine_new import ImprovedMADEngineNew

class MultiAgentEvaluator:
    """Evluation for different loop modes, firstloop or BaseMAD.
    Firstloop mode: Runs planner -> skeptic -> reasoner/SQL -> execution accuracy.
    BaseMAD mode: Runs planner -> debate between defender and skeptic -> reasoner/SQL -> execution accuracy.
    """

    def __init__(self, config: PipelineConfig, engine: Optional[LoopEngine] = None):
        if config.dataset != "spider":
            raise ValueError("Currently only the Spider dataset is supported.")
        self.config = config
        # init_agents now returns (planner, defender_or_none, skeptic, reasoner)
        self.planner, self.defender, self.skeptic, self.reasoner = init_agents(config)

        if engine is not None:
            self.engine = engine
        else:
            # Choose engine based on loop_mode in config.
            loop_mode = getattr(config, "loop_mode", "first")
            if loop_mode == "BaseMAD":
                if self.defender is None:
                    raise ValueError("Defender agent must be initialized for BaseMAD mode.")
                self.engine = BaseMADEngine(
                    self.planner,
                    self.defender,
                    self.skeptic,
                    self.reasoner,
                    max_debate_rounds=getattr(config, "max_debate_rounds", 3),
                    enable_early_termination=getattr(config, "enable_early_termination", True),
                    min_debate_rounds=getattr(config, "min_debate_rounds", 2),
                )
            elif loop_mode == "heteroMAD":
                if self.defender is None:
                    raise ValueError("Defender agent must be initialized for heteroMAD mode.")
                self.engine = HeteroMADEngine(
                    self.planner,
                    self.defender,
                    self.skeptic,
                    self.reasoner,
                    max_debate_rounds=getattr(config, "max_debate_rounds", 3),
                    enable_early_termination=getattr(config, "enable_early_termination", True),
                    min_debate_rounds=getattr(config, "min_debate_rounds", 2),
                )
            elif loop_mode == "ImprovedMAD":
                if self.defender is None:
                    raise ValueError("Defender agent must be initialized for ImprovedMAD mode.")
                # ImprovedMAD: BaseMAD-style debate + contract+validator+two-candidate selection.
                # self.engine = ImprovedMADEngine(
                self.engine = ImprovedMADEngineNew(
                    self.planner,
                    self.defender,
                    self.skeptic,
                    max_debate_rounds=getattr(config, "max_debate_rounds", 3),
                    llm_config_reasoner=LLMConfig(temperature=config.reasoner_temperature, max_tokens=config.max_tokens),
                    llm_config_sqlgen=LLMConfig(temperature=0.0, max_tokens=config.max_tokens),
                    llm_config_oneshot=LLMConfig(temperature=0.0, max_tokens=config.max_tokens),
                )
            else:
                self.engine = FirstLoopEngine(self.planner, self.skeptic, self.reasoner)

        self.inputs_snapshot = snapshot_inputs(config)
        examples, tables, db_root, gold_sql = get_spider_paths(config.split)
        self.examples_path = Path(config.examples_path or examples)
        self.tables_path = Path(config.tables_path or tables)
        self.db_root = Path(config.db_root or db_root)
        self.gold_sql_path = Path(config.gold_sql_path or gold_sql)

        self.examples = read_examples(str(self.examples_path))
        self.gold_pairs = parse_gold_sql(str(self.gold_sql_path))
        if len(self.examples) != len(self.gold_pairs):
            n = min(len(self.examples), len(self.gold_pairs))
            self.examples = self.examples[:n]
            self.gold_pairs = self.gold_pairs[:n]

    def run(self):
        cfg = self.config
        start = max(cfg.start, 0)
        end = min(len(self.examples), start + cfg.max_examples) if cfg.max_examples > 0 else len(self.examples)

        total = 0
        correct = 0
        latency_sums = defaultdict(float)
        latency_counts = defaultdict(int)

        def record_latency(metric: str, value: float):
            latency_sums[metric] += value
            latency_counts[metric] += 1

        loop_mode = getattr(cfg, "loop_mode", "first")
        if loop_mode == "BaseMAD":
            loop_label = "BaseMAD"
        elif loop_mode == "heteroMAD":
            loop_label = "HeteroMAD"
        elif loop_mode == "ImprovedMAD":
            loop_label = "ImprovedMAD"
        else:
            loop_label = "First Loop"
        print(f"=== System {loop_label} Evaluation ===")
        print(f"Dataset split   : {cfg.split}")
        print(f"Examples path   : {self.examples_path}")
        print(f"Tables path     : {self.tables_path}")
        print(f"DB root         : {self.db_root}")
        print(f"Gold SQL path   : {self.gold_sql_path}")
        print(f"Logging to      : {cfg.log_path}")
        print(f"Processing range: [{start}, {end}) across {len(self.examples)} examples")


        for idx in range(start, end):
            example = self.examples[idx]
            gold_sql, gold_db_id = self.gold_pairs[idx]
            question = example.get("question", "")
            db_id = example.get("db_id") or gold_db_id

            schema_text = load_schema_text("spider", db_id, self.tables_path)
            log_entry: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "example_index": idx,
                "command_line": cfg.command_line,
                "question": question,
                "db_id": db_id,
                # "split": cfg.split,
                # "dataset": cfg.dataset,
                "inputs": self.inputs_snapshot,
                "latency": {},
            }

            db_path = resolve_db_path(self.db_root, db_id, cfg.split)
            if not db_path:
                log_entry["error"] = f"SQLite DB not found for {db_id}"
                log_entry["latency"]["total_sec"] = 0.0
                record_latency("total_sec", 0.0)
                update_json(cfg.log_path, log_entry)
                print(f"[WARN] DB missing for idx={idx}, db={db_id}")
                continue

            gold_sql_clean = (gold_sql or "").strip().rstrip(";")

            # Run the chosen loop engine (FirstLoopEngine or BaseMADEngine).
            try:
                agent_result = self.engine.run(
                    question,
                    schema_text,
                    db_path=str(db_path),
                    gold_sql=gold_sql_clean,
                )
            except LoopStageError as exc:
                partial = exc.partial or {}
                log_entry.update({k: v for k, v in partial.items() if k != "latency"})
                partial_lat = partial.get("latency", {})
                log_entry["latency"].update(partial_lat)
                for key in ("planner_sec", "skeptic_sec", "reasoner_decision_sec", "sqlgen_sec"):
                    if key in partial_lat:
                        record_latency(key, partial_lat[key])
                log_entry["error"] = str(exc)
                elapsed = exc.elapsed
                log_entry["latency"]["total_sec"] = elapsed
                record_latency("total_sec", elapsed)
                update_json(cfg.log_path, log_entry)
                print(f"[ERROR] {exc.stage} failed for idx={idx}, db={db_id}: {exc.original}")
                continue

            log_entry["plan"] = agent_result.plan
            log_entry["reasoner_decision"] = agent_result.reasoner_decision
            log_entry["final_sql"] = agent_result.final_sql
            log_entry["latency"].update(agent_result.latency)
            if loop_mode in ("BaseMAD", "heteroMAD"):
                # In BaseMAD/heteroMAD, skeptic_feedback holds the debate summary, and debug carries per-round traces.
                log_entry["debate_summary"] = agent_result.skeptic_feedback
                if getattr(agent_result, "debug", None):
                    if loop_mode == "heteroMAD":
                        log_entry["heteromad_debug"] = agent_result.debug
                    else:
                        log_entry["basemad_debug"] = agent_result.debug
                # For heteroMAD: log which LLMs were chosen for each agent
                if loop_mode == "heteroMAD" and getattr(agent_result, "llm_choices", None):
                    log_entry["llm_choices"] = agent_result.llm_choices
            else:
                log_entry["skeptic_feedback"] = agent_result.skeptic_feedback
            base_total = agent_result.latency.get("total_sec", 0.0)
            final_sql = agent_result.final_sql

            per_example_metrics = (
                "planner_sec",
                "defender_total_sec",
                "skeptic_total_sec",
                "debate_total_sec",
                "debate_sec",
                "contract_reasoner_sec",
                "contract_validate_sec",
                "sqlgen_contract_sec",
                "sqlgen_oneshot_sec",
                "choose_judge_sec",
                "choose_sec",
                "skeptic_sec",
                "reasoner_decision_sec",
                "sqlgen_sec",
                "execution_sec",
                "total_sec",
            )
            for key in per_example_metrics:
                if key in agent_result.latency:
                    record_latency(key, agent_result.latency[key])

            exec_start = time.perf_counter()
            try:
                gen_rows = exec_sql(db_path, final_sql) if final_sql else []
            except Exception as exc:
                gen_rows = []
                log_entry["execution_error"] = f"Generated SQL failed: {exc}"
            try:
                gold_rows = exec_sql(db_path, gold_sql_clean) if gold_sql_clean else []
            except Exception as exc:
                gold_rows = []
                log_entry["gold_execution_error"] = f"Gold SQL failed: {exc}"
            exec_latency = time.perf_counter() - exec_start

            # Log execution results for debugging/inspection
            # log_entry["execution_result"] = gen_rows
            # log_entry["gold_execution_result"] = gold_rows

            total_time = base_total + exec_latency
            log_entry["latency"]["execution_sec"] = exec_latency
            log_entry["latency"]["total_sec"] = total_time
            record_latency("execution_sec", exec_latency)
            record_latency("total_sec", total_time)

            total += 1

            # Compare results ignoring row order and column ordering differences
            gen_counter = canonicalize_rows(gen_rows or [])
            gold_counter = canonicalize_rows(gold_rows or [])
            hit = gen_counter == gold_counter
            correct += int(hit)
            log_entry["exec_match"] = hit
            log_entry["accuracy_so_far"] = correct / total if total else 0.0

            if cfg.save_schema:
                log_entry["schema_text"] = schema_text

            update_json(cfg.log_path, log_entry)

            if (idx - start + 1) % 10 == 0:
                print(f"[{idx + 1}/{end}] acc={correct / total:.4f}")

        print(f"\nFinal Execution Accuracy: {correct}/{total} = {correct / total if total else 0:.4f}")

        if latency_counts:
            print("\nLatency summary (total | avg per example | examples):")
            if loop_mode in ("BaseMAD", "heteroMAD"):
                ordered_metrics = [
                    ("planner_sec", "Planner"),
                    ("defender_total_sec", "Defender (total)"),
                    ("skeptic_total_sec", "Skeptic (total)"),
                    ("debate_total_sec", "Debate (total)"),
                    ("reasoner_decision_sec", "Reasoner decision"),
                    ("sqlgen_sec", "SQL generation"),
                    ("execution_sec", "Execution"),
                    ("total_sec", "Whole system"),
                ]
            elif loop_mode == "ImprovedMAD":
                ordered_metrics = [
                    ("planner_sec", "Planner"),
                    ("debate_sec", "Debate"),
                    ("contract_reasoner_sec", "Contract reasoner"),
                    ("contract_validate_sec", "Contract validate"),
                    ("sqlgen_contract_sec", "SQLGen (contract)"),
                    ("sqlgen_oneshot_sec", "SQLGen (oneshot)"),
                    ("choose_judge_sec", "Choose (judge)"),
                    ("execution_sec", "Execution"),
                    ("total_sec", "Whole system"),
                ]
            else:
                ordered_metrics = [
                    ("planner_sec", "Planner"),
                    ("skeptic_sec", "Skeptic"),
                    ("reasoner_decision_sec", "Reasoner decision"),
                    ("sqlgen_sec", "SQL generation"),
                    ("execution_sec", "Execution"),
                    ("total_sec", "Whole system"),
                ]
            for key, label in ordered_metrics:
                count = latency_counts.get(key, 0)
                if not count:
                    continue
                total_latency = latency_sums.get(key, 0.0)
                avg_latency = total_latency / count if count else 0.0
                print(f"  {label:<18}: {total_latency:8.2f}s | {avg_latency:8.2f}s | {count:3d}")

