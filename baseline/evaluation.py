import os
import sys
import json
import sqlite3
import time
from typing import List, Tuple, Any, Iterable
import argparse
from pathlib import Path
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.dataloader import load_spider
from baseline.planneragent import PlannerGraph
from baseline.baseagent import BaseGraph
from baseline.state import PlannerState, BaseState
from baseline.llm import LLMConfig
from baseline.exec_match import canonicalize_rows, get_spider_paths, get_dataset_paths, read_examples, parse_gold_sql, resolve_db_path, exec_sql
# Reuse utilities from run_planner to get schema text and write logs
from baseline.run_planner import get_table_paths 
from baseline.run_planner import load_schema_text 
from baseline.run_planner import save_log 
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Compute execution accuracy against gold SQL.")
    parser.add_argument("--dataset", type=str, default="spider", choices=["spider", "bird", "BIRD"], help="Dataset to evaluate: spider or bird")
    parser.add_argument("--split", type=str, default="test", help="Dataset split: test or dev")
    parser.add_argument("--examples_path", type=str, help="Override path to examples JSON")
    parser.add_argument("--tables_path", type=str, help="Override path to tables JSON")
    parser.add_argument("--db_root", type=str, help="Override path to DB root folder")
    parser.add_argument("--gold_sql_path", type=str, help="Override path to split_gold.sql")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=10**9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--baseline", choices=["planner", "baseagent"], default="planner", help="Choose which baseline agent to evaluate 'planner' or 'baseagent'")
    # Logging options consistent with run_planner
    parser.add_argument("--log_path", type=str, help="Path to save log file (JSON). Default: baseline_logs/planner_log.json")
    parser.add_argument("--save_schema", action="store_true", help="Include schema_text in the log file")
    args = parser.parse_args()

    # Normalize dataset name
    dataset = args.dataset.lower()

    # Resolve paths
    default_examples, default_tables, default_db_root, default_gold = get_dataset_paths(dataset, args.split)
    examples_path = args.examples_path or default_examples
    db_root = args.db_root or default_db_root
    gold_sql_path = args.gold_sql_path or default_gold

    # Use run_planner's table path resolver to match its behavior
    tables_path = get_table_paths(dataset, args.split, args.tables_path)

    # Load data
    print("=== Resolved Paths ===")
    print(f"examples_path: {examples_path}")
    print(f"tables_path : {tables_path}")
    print(f"db_root     : {db_root}")
    print(f"gold_sql    : {gold_sql_path}")

    examples = read_examples(examples_path)
    gold_pairs = parse_gold_sql(gold_sql_path)

    print(f"examples count: {len(examples)}")
    print(f"gold count    : {len(gold_pairs)}")

    # Enforce 1:1 alignment as far as possible
    if len(gold_pairs) != len(examples):
        min_len = min(len(gold_pairs), len(examples))
        print(
            f"Warning: examples ({len(examples)}) and gold ({len(gold_pairs)}) differ. "
            f"Evaluating first {min_len} aligned items."
        )
        examples = examples[:min_len]
        gold_pairs = gold_pairs[:min_len]

    # Load schema metadata (for run_planner.load_schema_text)
    # Note: This is kept for parity but not strictly needed for evaluation
    if dataset == "spider":
        spider_tables = load_spider(tables_path)
    # BIRD tables loading is handled by load_schema_text when needed

    # Select baseline agent
    config = LLMConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    if args.baseline == "planner":
        graph = PlannerGraph(config)
        state_cls = PlannerState
        expects_plan = True
    else:
        graph = BaseGraph(config)
        state_cls = BaseState
        expects_plan = False

    # Resolve default log path like run_planner
    if args.log_path:
        log_path = args.log_path
    else:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        default_log = "planner_log.json" if args.baseline == "planner" else "baseagent_log.json"
        log_path = os.path.join(log_dir, default_log)

    command_line = " ".join(sys.argv)

    start = max(0, args.start)
    end = min(len(examples), start + max(0, args.max_examples))
    total = 0
    correct = 0
    gen_latency_sum = 0.0
    exec_latency_sum = 0.0
    total_latency_sum = 0.0

    print(f"\n=== Starting Evaluation ===")
    print(f"Processing examples from index {start} to {end-1} (total: {end-start} examples)")
    print(f"Baseline: {args.baseline}")
    print(f"Log path: {log_path}")
    print(f"Progress will be printed every 10 examples")
    print(f"Note: Each LLM call may take up to 120 seconds (timeout). Please be patient...\n")

    for idx in range(start, end):
        total_start = time.perf_counter()
        ex = examples[idx]
        question = ex.get("question", "")
        db_id = ex.get("db_id", "")
        gold_sql, gold_db_id = gold_pairs[idx]

        # Use the example's db_id for schema and db path; gold's db_id is a sanity check
        active_db_id = db_id or gold_db_id

        # Get schema text via run_planner utility (to match planner usage)
        schema_text = load_schema_text(dataset, active_db_id, tables_path)

        # Invoke selected agent to generate SQL for the current question
        if (idx - start) % 10 == 0 or idx == start:
            print(f"[{idx+1}/{end}] Processing example {idx} (db_id: {active_db_id})...", flush=True)
        
        state = state_cls(question=question, db_id=active_db_id, schema_text=schema_text)
        out_state = None
        gen_latency = 0.0
        try:
            if (idx - start) % 10 == 0 or idx == start:
                print(f"  → Calling LLM to generate SQL...", flush=True)
            start_time = time.perf_counter()
            out_dict = graph.invoke(state)
            gen_latency = time.perf_counter() - start_time
            if (idx - start) % 10 == 0 or idx == start:
                print(f"  → LLM call completed in {gen_latency:.2f}s", flush=True)
            out_state = state_cls(**out_dict)
            gen_sql = (out_state.sql or "").strip().rstrip(";")
        except Exception as e:
            print(f"[ERROR] Failed to generate plan/SQL for question '{question[:50]}...': {e}", flush=True)
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
            gen_sql = ""
            out_state = state_cls(question=question, db_id=active_db_id, schema_text=schema_text)

        plan = getattr(out_state, "plan", {}) if expects_plan and out_state else {}
        log_inputs = {
            "question": question,
            "db_id": active_db_id,
            "dataset": dataset,
            "split": args.split,
            "baseline": args.baseline,
            "example_index": idx,
            "latency": {
                "generation_sec": gen_latency,
            },
        }

        # Execute both queries
        db_path = resolve_db_path(db_root, active_db_id, args.split, dataset)
        if not db_path:
            print(f"[WARN] SQLite not found for db_id '{active_db_id}'. Checked '{db_root}' and standard Spider locations.", flush=True)
            total_latency = time.perf_counter() - total_start
            log_inputs["latency"]["execution_sec"] = 0.0
            log_inputs["latency"]["total_sec"] = total_latency
            try:
                if (idx - start) % 10 == 0 or idx == start:
                    print(f"  → Saving log...", flush=True)
                save_log(
                    log_path=log_path,
                    command_line=command_line,
                    inputs=log_inputs,
                    plan=plan,
                    sql=gen_sql,
                    schema_text=schema_text if args.save_schema else None,
                    latency_sec=gen_latency,
                )
                if (idx - start) % 10 == 0 or idx == start:
                    print(f"  → Log saved", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to save log: {e}", flush=True)
            total += 1
            continue

        gold_sql_clean = (gold_sql or "").strip().rstrip(";")
        if (idx - start) % 10 == 0 or idx == start:
            print(f"  → Executing SQL queries...", flush=True)
        exec_start = time.perf_counter()
        try:
            gen_rows = exec_sql(db_path, gen_sql) if gen_sql else []
        except TimeoutError as e:
            if (idx - start) % 10 == 0 or idx == start:
                print(f"  → TIMEOUT: Generated SQL execution exceeded 30s timeout: {e}", flush=True)
                print(f"  → Generated SQL: {gen_sql[:200]}...", flush=True)
            gen_rows = []
        except Exception as e:
            if (idx - start) % 10 == 0 or idx == start:
                print(f"  → Warning: Generated SQL execution failed: {e}", flush=True)
                print(f"  → Generated SQL: {gen_sql[:200]}...", flush=True)
            gen_rows = []
        try:
            gold_rows = exec_sql(db_path, gold_sql_clean) if gold_sql_clean else []
        except TimeoutError as e:
            if (idx - start) % 10 == 0 or idx == start:
                print(f"  → TIMEOUT: Gold SQL execution exceeded 30s timeout: {e}", flush=True)
                print(f"  → Gold SQL: {gold_sql_clean[:200]}...", flush=True)
            gold_rows = []
        except Exception as e:
            if (idx - start) % 10 == 0 or idx == start:
                print(f"  → Warning: Gold SQL execution failed: {e}", flush=True)
                print(f"  → Gold SQL: {gold_sql_clean[:200]}...", flush=True)
            gold_rows = []

        exec_latency = time.perf_counter() - exec_start
        total_latency = time.perf_counter() - total_start
        log_inputs["latency"]["execution_sec"] = exec_latency
        log_inputs["latency"]["total_sec"] = total_latency
        if (idx - start) % 10 == 0 or idx == start:
            print(f"  → SQL execution completed in {exec_latency:.2f}s", flush=True)

        # Compare results ignoring order (SQL doesn't guarantee order without ORDER BY)
        # and column positions (agent might alias/select columns in different order)
        gen_counter = canonicalize_rows(gen_rows) if gen_rows else Counter()
        gold_counter = canonicalize_rows(gold_rows) if gold_rows else Counter()
        hit = gen_counter == gold_counter
        
        log_inputs["exec_match"] = hit

        try:
            if (idx - start) % 10 == 0 or idx == start:
                print(f"  → Saving log...", flush=True)
            save_log(
                log_path=log_path,
                command_line=command_line,
                inputs=log_inputs,
                plan=plan,
                sql=gen_sql,
                schema_text=schema_text if args.save_schema else None,
                latency_sec=gen_latency,
            )
            if (idx - start) % 10 == 0 or idx == start:
                print(f"  → Example {idx} completed (match: {hit}, total time: {total_latency:.2f}s)", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to save log: {e}", flush=True)

        total += 1
        gen_latency_sum += gen_latency
        exec_latency_sum += exec_latency
        total_latency_sum += total_latency
        correct += int(hit)

        if (idx - start + 1) % 25 == 0:
            print(f"[{idx + 1}/{end}] acc={correct/total:.4f}")

    print(f"Execution Accuracy ({total} examples): {correct/total if total else 0.0:.4f}  (correct: {correct})")
    if total:
        print(
            f"Average Latencies (s): generation={gen_latency_sum/total:.3f}, "
            f"execution={exec_latency_sum/total:.3f}, total={total_latency_sum/total:.3f}"
        )
        print(
            f"Total Latencies (s): generation={gen_latency_sum:.3f}, "
            f"execution={exec_latency_sum:.3f}, total={total_latency_sum:.3f}"
        )

if __name__ == "__main__":
    main()