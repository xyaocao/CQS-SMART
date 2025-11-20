import json
import time
from pathlib import Path
from typing import Any, Dict, List
import sys
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.run_planner import load_schema_text  
from baseline.evaluation import get_spider_paths, read_examples  
from pipeline_utils import PipelineConfig, update_json, init_agents, snapshot_inputs

class SinglerunPipeline:
    """Answers a single question by using the first-loop agents: planner -> skeptic -> reasoner (with sqlgen)."""
    def __init__(self, config: PipelineConfig):
        if config.input_mode != "single":
            raise ValueError("SinglerunPipeline requires input_mode='single'")
        self.config = config
        self.planner, self.skeptic, self.reasoner = init_agents(config)
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
        total_start = time.perf_counter()
        try:
            plan, plan_latency = self.planner.run(question, schema_text)
            log_entry["plan"] = plan
            log_entry["latency"]["planner_sec"] = plan_latency
            # log_entry["planner_raw"] = planner_raw
        except Exception as exc:
            log_entry["error"] = f"Planner failure: {exc}"
            log_entry["latency"]["total_sec"] = time.perf_counter() - total_start
            update_json(cfg.log_path, log_entry)
            raise
        try: 
            feedback, skeptic_latency = self.skeptic.run(question, schema_text, plan)
            log_entry["skeptic_feedback"] = feedback
            log_entry["latency"]["skeptic_sec"] = skeptic_latency
            # log_entry["skeptic_raw"] = skeptic_raw
        except Exception as exc:
            log_entry["error"] = f"Skeptic failure: {exc}"
            log_entry["latency"]["total_sec"] = time.perf_counter() - total_start
            update_json(cfg.log_path, log_entry)
            raise
        try:
            decision, decision_latency, final_sql, sql_latency = self.reasoner.run(
                question, schema_text, plan, feedback
            )
            log_entry["reasoner_decision"] = decision
            log_entry["latency"]["reasoner_decision_sec"] = decision_latency
            # log_entry["reasoner_raw"] = decision_raw
            log_entry["final_sql"] = final_sql
            log_entry["latency"]["sqlgen_sec"] = sql_latency
            # log_entry["sqlgen_raw"] = sql_raw
        except Exception as exc:
            log_entry["error"] = f"Reasoner failure: {exc}"
            log_entry["latency"]["total_sec"] = time.perf_counter() - total_start
            update_json(cfg.log_path, log_entry)
            raise

        total_time = time.perf_counter() - total_start
        log_entry["latency"]["execution_sec"] = 0.0
        log_entry["latency"]["total_sec"] = total_time

        if cfg.save_schema:
            log_entry["schema_text"] = schema_text

        update_json(cfg.log_path, log_entry)

        print("\n=== PLAN ===")
        print(json.dumps(plan, ensure_ascii=False))
        print("\n=== SKEPTIC FEEDBACK ===")
        print(json.dumps(log_entry["skeptic_feedback"], ensure_ascii=False))
        print("\n=== REASONER DECISION ===")
        print(json.dumps(decision, ensure_ascii=False))
        print("\n=== SQL ===")
        print(final_sql.strip())

class BatchPipeline:
    """Processes a batch of examples using the first-loop agents: planner -> skeptic -> reasoner (with sqlgen)."""
    def __init__(self, config: PipelineConfig):
        if config.input_mode != "batch":
            raise ValueError("BatchPipeline requires input_mode='batch'")
        self.config = config
        self.planner, self.skeptic, self.reasoner = init_agents(config)
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
            total_start = time.perf_counter()

            try:
                plan, plan_latency = self.planner.run(question, schema_text)
                log_entry["plan"] = plan
                log_entry["latency"]["planner_sec"] = plan_latency
                # log_entry["planner_raw"] = planner_raw
            except Exception as exc:
                log_entry["error"] = f"Planner failure: {exc}"
                log_entry["latency"]["total_sec"] = time.perf_counter() - total_start
                update_json(cfg.log_path, log_entry)
                print(f"[ERROR] Planner failed for idx={idx}, db={db_id}: {exc}")
                continue
            try:
                feedback, skeptic_latency = self.skeptic.run(question, schema_text, plan)
                log_entry["skeptic_feedback"] = feedback
                log_entry["latency"]["skeptic_sec"] = skeptic_latency
                # log_entry["skeptic_raw"] = skeptic_raw
            except Exception as exc:
                log_entry["error"] = f"Skeptic failure: {exc}"
                log_entry["latency"]["total_sec"] = time.perf_counter() - total_start
                update_json(cfg.log_path, log_entry)
                print(f"[ERROR] Skeptic failed for idx={idx}, db={db_id}: {exc}")
                continue

            try:
                decision, decision_latency, final_sql, sql_latency = self.reasoner.run(question, schema_text, plan, feedback)
                log_entry["reasoner_decision"] = decision
                log_entry["latency"]["reasoner_decision_sec"] = decision_latency
                # log_entry["reasoner_raw"] = decision_raw
                log_entry["final_sql"] = final_sql
                log_entry["latency"]["sqlgen_sec"] = sql_latency
                # log_entry["sqlgen_raw"] = sql_raw
            except Exception as exc:
                log_entry["error"] = f"Reasoner failure: {exc}"
                log_entry["latency"]["total_sec"] = time.perf_counter() - total_start
                update_json(cfg.log_path, log_entry)
                print(f"[ERROR] Reasoner failed for idx={idx}, db={db_id}: {exc}")
                continue

            total_time = time.perf_counter() - total_start
            log_entry["latency"]["execution_sec"] = 0.0
            log_entry["latency"]["total_sec"] = total_time

            if cfg.save_schema:
                log_entry["schema_text"] = schema_text

            update_json(cfg.log_path, log_entry)

            print(f"\n=== Example {idx} | db_id={db_id} ===")
            print("Question:", question)
            print("Plan:", json.dumps(plan, ensure_ascii=False))
            print("Skeptic Feedback:", json.dumps(feedback, ensure_ascii=False))
            print("Reasoner Decision:", json.dumps(decision, ensure_ascii=False))
            print("SQL:", final_sql.strip())

        print(f"\nProcessed {max(0, end - start)} examples.")

