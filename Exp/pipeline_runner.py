import json
from pathlib import Path
from typing import Any, Dict, List
import sys
from datetime import datetime
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.run_planner import load_schema_text  
from baseline.evaluation import get_spider_paths, read_examples 
from pipeline_utils import PipelineConfig, update_json, init_agents, snapshot_inputs
from loop_engines import FirstLoopEngine, BaseMADEngine, LoopStageError

class SinglerunPipeline:
    """Answers a single question using either first-loop or second-loop agents.

    loop_mode in PipelineConfig controls behaviour:
      - 'first':  Planner -> Skeptic -> Reasoner (with sqlgen)
      - 'BaseMAD': Planner -> Debate between Defender and Skeptic -> Reasoner makes the final decision (with sqlgen)
    """
    def __init__(self, config: PipelineConfig):
        if config.input_mode != "single":
            raise ValueError("SinglerunPipeline requires input_mode='single'")
        self.config = config
        self.planner, self.defender, self.skeptic, self.reasoner = init_agents(config)
        # Choose engine based on loop_mode
        if config.loop_mode == "BaseMAD":
            if self.defender is None:
                raise ValueError("Defender agent must be initialized for BaseMAD mode.")
            self.engine = BaseMADEngine(
                self.planner,
                self.defender,
                self.skeptic,
                self.reasoner,
                max_debate_rounds=config.max_debate_rounds,
                enable_early_termination=getattr(config, "enable_early_termination", True),
                min_debate_rounds=getattr(config, "min_debate_rounds", 2),
            )
        else:
            self.engine = FirstLoopEngine(self.planner, self.skeptic, self.reasoner)
        self.inputs_snapshot = snapshot_inputs(config)
        examples, tables, db_root, _ = get_spider_paths(config.split)
        self.tables_path = Path(config.tables_path or tables)
        self.db_root = Path(config.db_root or db_root)

    def run(self):
        cfg = self.config
        question = cfg.question or ""
        db_id = cfg.db_id or ""
        schema_text = load_schema_text(cfg.dataset, db_id, str(self.tables_path))
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "command_line": cfg.command_line,
            "question": question,
            "db_id": db_id,
            # "split": cfg.split,
            # "dataset": cfg.dataset,
            "inputs": self.inputs_snapshot,
            "latency": {},
        }

        extra_kwargs: Dict[str, Any] = {}
       
        try:
            agent_result = self.engine.run(question, schema_text, **extra_kwargs)
        except LoopStageError as exc:
            partial = exc.partial or {}
            log_entry.update({k: v for k, v in partial.items() if k != "latency"})
            log_entry["latency"].update(partial.get("latency", {}))
            log_entry["error"] = str(exc)
            log_entry["latency"]["total_sec"] = exc.elapsed
            update_json(cfg.log_path, log_entry)
            raise

        log_entry["plan"] = agent_result.plan
        log_entry["reasoner_decision"] = agent_result.reasoner_decision
        log_entry["final_sql"] = agent_result.final_sql
        log_entry["latency"].update(agent_result.latency)

        if self.config.loop_mode == "BaseMAD":
            # In BaseMAD, skeptic_feedback holds the debate summary, and debug carries per-round traces.
            log_entry["debate_summary"] = agent_result.skeptic_feedback
            if getattr(agent_result, "debug", None):
                log_entry["basemad_debug"] = agent_result.debug
        else:
            # First-loop behaviour: skeptic_feedback is the single skeptic pass.
            log_entry["skeptic_feedback"] = agent_result.skeptic_feedback

        if cfg.save_schema:
            log_entry["schema_text"] = schema_text

        update_json(cfg.log_path, log_entry)

        print("\n=== PLAN (s0) ===")
        print(json.dumps(agent_result.plan, ensure_ascii=False))

        if self.config.loop_mode == "BaseMAD":
            debug = getattr(agent_result, "debug", {}) or {}
            debate_rounds = debug.get("debate_rounds", [])
            print("\n=== DEBATE: DEFENDER vs SKEPTIC ===")
            actual_rounds = debug.get("actual_rounds", len(debate_rounds))
            max_rounds = debug.get("max_debate_rounds")
            termination_reason = debug.get("termination_reason", "max_rounds_reached")
            print(f"Rounds run: {actual_rounds} (max={max_rounds})")
            if debug.get("early_terminated"):
                print(f"Early termination: {termination_reason}")
            for r in debate_rounds:
                print(f"\n--- Round {r.get('round')} ---")
                print("Defender View:", json.dumps(r.get("defender_view", {}), ensure_ascii=False))
                print("Skeptic View:", json.dumps(r.get("skeptic_view", {}), ensure_ascii=False))
                print("Consensus:", r.get("consensus"))
            print("\n=== FINAL REASONER DECISION ===")
            print(json.dumps(agent_result.reasoner_decision, ensure_ascii=False))
            print("\n=== FINAL SQL ===")
            print(agent_result.final_sql.strip())
        else:
            print("\n=== SKEPTIC FEEDBACK ===")
            print(json.dumps(agent_result.skeptic_feedback, ensure_ascii=False))
            print("\n=== REASONER DECISION ===")
            print(json.dumps(agent_result.reasoner_decision, ensure_ascii=False))
            print("\n=== SQL ===")
            print(agent_result.final_sql.strip())

class BatchPipeline:
    """Processes a batch of examples using either first-loop or BaseMAD agents.

    loop_mode in PipelineConfig controls behaviour (see SinglerunPipeline).
    """
    def __init__(self, config: PipelineConfig):
        if config.input_mode != "batch":
            raise ValueError("BatchPipeline requires input_mode='batch'")
        self.config = config
        self.planner, self.defender, self.skeptic, self.reasoner = init_agents(config)
        if config.loop_mode == "BaseMAD":
            if self.defender is None:
                raise ValueError("Defender agent must be initialized for BaseMAD mode.")
            self.engine = BaseMADEngine(
                self.planner,
                self.defender,
                self.skeptic,
                self.reasoner,
                max_debate_rounds=config.max_debate_rounds,
                enable_early_termination=getattr(config, "enable_early_termination", True),
                min_debate_rounds=getattr(config, "min_debate_rounds", 2),
            )
        else:
            self.engine = FirstLoopEngine(self.planner, self.skeptic, self.reasoner)
        self.inputs_snapshot = snapshot_inputs(config)
        examples, tables, db_root, gold_sql = get_spider_paths(config.split)
        self.examples_path = Path(config.examples_path or examples)
        self.tables_path = Path(config.tables_path or tables)
        self.db_root = Path(config.db_root or db_root)
        self.examples: List[Dict[str, Any]] = read_examples(str(self.examples_path))

    def run(self):
        cfg = self.config
        start = max(cfg.start, 0)
        end = min(len(self.examples), start + cfg.max_examples) if cfg.max_examples > 0 else len(self.examples)

        loop_label = "BaseMAD" if cfg.loop_mode == "BaseMAD" else "First Loop"
        print(f"=== System {loop_label} ===")
        print(f"Dataset split : {cfg.split}")
        print(f"Examples path : {self.examples_path}")
        print(f"Tables path   : {self.tables_path}")
        print(f"Logging to    : {cfg.log_path}")
        print(f"Range         : ({start}, {end}) / {len(self.examples)}")

        for idx in range(start, end):
            example = self.examples[idx]
            question = example.get("question", "")
            db_id = example.get("db_id", "")
            schema_text = load_schema_text(cfg.dataset, db_id, str(self.tables_path))
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

            extra_kwargs: Dict[str, Any] = {}

            try:
                agent_result = self.engine.run(question, schema_text, **extra_kwargs)
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
            log_entry["reasoner_decision"] = agent_result.reasoner_decision
            log_entry["final_sql"] = agent_result.final_sql
            log_entry["latency"].update(agent_result.latency)

            if cfg.loop_mode == "BaseMAD":
                log_entry["debate_summary"] = agent_result.skeptic_feedback
                if getattr(agent_result, "debug", None):
                    log_entry["basemad_debug"] = agent_result.debug
            else:
                log_entry["skeptic_feedback"] = agent_result.skeptic_feedback

            if cfg.save_schema:
                log_entry["schema_text"] = schema_text

            update_json(cfg.log_path, log_entry)

            print(f"\n=== Example {idx} | db_id={db_id} ===")
            print("Question:", question)
            print("Plan:", json.dumps(agent_result.plan, ensure_ascii=False))

            if cfg.loop_mode == "BaseMAD":
                debug = getattr(agent_result, "debug", {}) or {}
                debate_rounds = debug.get("debate_rounds", [])
                actual_rounds = debug.get("actual_rounds", len(debate_rounds))
                termination_reason = debug.get("termination_reason", "max_rounds_reached")
                print(f"Debate Rounds: {actual_rounds} (max={debug.get('max_debate_rounds')})")
                if debug.get("early_terminated"):
                    print(f"  Early termination: {termination_reason}")
                for r in debate_rounds:
                    print(f"  - Round {r.get('round')}, consensus={r.get('consensus')}")
                print("Reasoner Decision:", json.dumps(agent_result.reasoner_decision, ensure_ascii=False))
                print("SQL:", agent_result.final_sql.strip())
            else:
                print("Skeptic Feedback:", json.dumps(agent_result.skeptic_feedback, ensure_ascii=False))
                print("Reasoner Decision:", json.dumps(agent_result.reasoner_decision, ensure_ascii=False))
                print("SQL:", agent_result.final_sql.strip())

        print(f"\nProcessed {max(0, end - start)} examples.")
