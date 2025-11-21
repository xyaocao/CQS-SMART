import json
from pathlib import Path
from typing import Any, Dict, List
import sys
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.run_planner import load_schema_text  
from baseline.evaluation import get_spider_paths, read_examples  
from pipeline_utils import PipelineConfig, update_json, init_agents, snapshot_inputs
from loop_engines import FirstLoopEngine, LoopStageError

class SinglerunPipeline:
    """Answers a single question by using the first-loop agents: planner -> skeptic -> reasoner (with sqlgen)."""
    def __init__(self, config: PipelineConfig):
        if config.input_mode != "single":
            raise ValueError("SinglerunPipeline requires input_mode='single'")
        self.config = config
        self.planner, self.skeptic, self.reasoner = init_agents(config)
        self.engine = FirstLoopEngine(self.planner, self.skeptic, self.reasoner)
        self.inputs_snapshot = snapshot_inputs(config)
        examples, tables, db_root, gold_sql = get_spider_paths(config.split)
        self.tables_path = Path(config.tables_path or tables)
        self.db_root = Path(config.db_root or db_root)

    def run(self):
        cfg = self.config
        question = cfg.question or ""
        db_id = cfg.db_id or ""
        schema_text = load_schema_text(cfg.dataset, db_id, str(self.tables_path))
        log_entry: Dict[str, Any] = {
            "timestamp": cfg.timestamp,
            "command_line": cfg.command_line,
            "question": question,
            "db_id": db_id,
            # "split": cfg.split,
            # "dataset": cfg.dataset,
            "inputs": self.inputs_snapshot,
            "latency": {},
        }
        try:
            agent_result = self.engine.run(question, schema_text)
        except LoopStageError as exc:
            partial = exc.partial or {}
            log_entry.update({k: v for k, v in partial.items() if k != "latency"})
            log_entry["latency"].update(partial.get("latency", {}))
            log_entry["error"] = str(exc)
            log_entry["latency"]["total_sec"] = exc.elapsed
            update_json(cfg.log_path, log_entry)
            raise

        log_entry["plan"] = agent_result.plan
        log_entry["skeptic_feedback"] = agent_result.skeptic_feedback
        log_entry["reasoner_decision"] = agent_result.reasoner_decision
        log_entry["final_sql"] = agent_result.final_sql
        log_entry["latency"].update(agent_result.latency)

        if cfg.save_schema:
            log_entry["schema_text"] = schema_text

        update_json(cfg.log_path, log_entry)

        print("\n=== PLAN ===")
        print(json.dumps(agent_result.plan, ensure_ascii=False))
        print("\n=== SKEPTIC FEEDBACK ===")
        print(json.dumps(log_entry["skeptic_feedback"], ensure_ascii=False))
        print("\n=== REASONER DECISION ===")
        print(json.dumps(agent_result.reasoner_decision, ensure_ascii=False))
        print("\n=== SQL ===")
        print(agent_result.final_sql.strip())

class BatchPipeline:
    """Processes a batch of examples using the first-loop agents: planner -> skeptic -> reasoner (with sqlgen)."""
    def __init__(self, config: PipelineConfig):
        if config.input_mode != "batch":
            raise ValueError("BatchPipeline requires input_mode='batch'")
        self.config = config
        self.planner, self.skeptic, self.reasoner = init_agents(config)
        self.engine = FirstLoopEngine(self.planner, self.skeptic, self.reasoner)
        self.inputs_snapshot = snapshot_inputs(config)
        examples, tables, db_root, gold_sql = get_spider_paths(config.split)
        self.examples_path = Path(config.examples_path or examples)
        self.tables_path = Path(config.tables_path or tables)
        self.examples: List[Dict[str, Any]] = read_examples(str(self.examples_path))

    def run(self):
        cfg = self.config
        start = max(cfg.start, 0)
        end = min(len(self.examples), start + cfg.max_examples) if cfg.max_examples > 0 else len(self.examples)

        print("=== Multi-Agent First Loop ===")
        print(f"Dataset split : {cfg.split}")
        print(f"Examples path : {self.examples_path}")
        print(f"Tables path   : {self.tables_path}")
        print(f"Logging to    : {cfg.log_path}")
        print(f"Range         : {start}, {end}) / {len(self.examples)}")

        for idx in range(start, end):
            example = self.examples[idx]
            question = example.get("question", "")
            db_id = example.get("db_id", "")
            schema_text = load_schema_text(cfg.dataset, db_id, str(self.tables_path))
            log_entry: Dict[str, Any] = {
                "timestamp": cfg.timestamp,
                "example_index": idx,
                "command_line": cfg.command_line,
                "question": question,
                "db_id": db_id,
                # "split": cfg.split,
                # "dataset": cfg.dataset,
                "inputs": self.inputs_snapshot,
                "latency": {},
            }
            try:
                agent_result = self.engine.run(question, schema_text)
            except LoopStageError as exc:
                partial = exc.partial or {}
                log_entry.update({k: v for k, v in partial.items() if k != "latency"})
                log_entry["latency"].update(partial.get("latency", {}))
                log_entry["error"] = str(exc)
                log_entry["latency"]["total_sec"] = exc.elapsed
                update_json(cfg.log_path, log_entry)
                print(f"[ERROR] {exc.stage} failed for idx={idx}, db={db_id}: {exc.original}")
                continue

            log_entry["plan"] = agent_result.plan
            log_entry["skeptic_feedback"] = agent_result.skeptic_feedback
            log_entry["reasoner_decision"] = agent_result.reasoner_decision
            log_entry["final_sql"] = agent_result.final_sql
            log_entry["latency"].update(agent_result.latency)

            if cfg.save_schema:
                log_entry["schema_text"] = schema_text

            update_json(cfg.log_path, log_entry)

            print(f"\n=== Example {idx} | db_id={db_id} ===")
            print("Question:", question)
            print("Plan:", json.dumps(agent_result.plan, ensure_ascii=False))
            print("Skeptic Feedback:", json.dumps(agent_result.skeptic_feedback, ensure_ascii=False))
            print("Reasoner Decision:", json.dumps(agent_result.reasoner_decision, ensure_ascii=False))
            print("SQL:", agent_result.final_sql.strip())

        print(f"\nProcessed {max(0, end - start)} examples.")

