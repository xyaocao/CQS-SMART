import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, List
import sys
from pathlib import Path
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.dataloader import load_spider, get_schema_from_spider, load_bird, get_schema_from_bird
from baseline.planneragent import PlannerGraph
from baseline.state import PlannerState
from baseline.llm import LLMConfig

def project_root() -> str:
    """Get the root directory of the project."""
    # baseline is now at root level, so go up one directory from baseline
    return os.path.dirname(os.path.dirname(__file__))

def get_table_paths(dataset: str, split: str, tables_path: str | None) -> str:
    if tables_path:
        return tables_path
    root = project_root()
    if dataset == "spider":
        default_file = "tables.json" if split == "dev" else "test_tables.json"
        return os.path.join(root, "Data", "spider_data", default_file)
    elif dataset == "bird":
        return os.path.join(root, "Data", "BIRD", split, f"{split}_tables.json")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
def load_schema_text(dataset: str, db_id: str, tables_meta_path: str | None) -> str:
    if dataset == "spider":
        tables = load_spider(tables_meta_path)
        return get_schema_from_spider(tables, db_id)
    elif dataset == "bird":
        tables = load_bird(tables_meta_path)
        return get_schema_from_bird(tables, db_id)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def get_examples_path(dataset: str, split: str, examples_path: str | None) -> str:
    if examples_path:
        return examples_path

    root = project_root()
    if dataset == "spider":
        candidate = os.path.join(root, "Data", "spider_data", f"{split}.json")
    elif dataset == "bird":
        candidate = os.path.join(root, "Data", "BIRD", split, f"{split}.json")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Could not locate examples file for dataset '{dataset}' split '{split}'. "
            f"Tried: {candidate}"
        )
    return candidate


def build_schema_resolver(dataset: str, tables_meta_path: str | None) -> Callable[[str], str]:
    if dataset == "spider":
        tables = load_spider(tables_meta_path)
        # resolver = get_schema_from_spider(tables, db_id)
        def resolver(db_id: str) -> str:
            return get_schema_from_spider(tables, db_id)
    elif dataset == "bird":
        tables = load_bird(tables_meta_path)
        # resolver = get_schema_from_bird(tables, db_id)
        def resolver(db_id: str) -> str:
            return get_schema_from_bird(tables, db_id)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return resolver


def save_log(log_path: str, command_line: str, inputs: dict, plan: dict, sql: str, schema_text: str = None, latency_sec: float | None = None):
    """Save the execution log to a JSON file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "command_line": command_line,
        "inputs": inputs,
        # "question": question,
        # "db_id": db_id,
        # "dataset": dataset,
        # "split": split,
        "plan": plan,
        "sql": sql,
    }
    if latency_sec is not None:
        log_entry["latency_sec"] = latency_sec
    
    if schema_text:
        log_entry["schema_text"] = schema_text
    
    # Create log directory if it doesn't exist
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing logs if file exists
    logs = []
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []
    
    # Append new log entry
    logs.append(log_entry)
    
    # Save updated logs
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    
    # print(f"\n=== Log saved to: {log_path} ===")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, nargs="?", help="Natural language question (required for single mode)")
    parser.add_argument("db_id", type=str, nargs="?", help="Database id (folder name) (required for single mode)")
    parser.add_argument("--dataset", choices=["bird", "spider"], default="spider")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--tables_path", type=str, help="Override path to tables metadata json")
    parser.add_argument("--examples_path", type=str, help="Override path to dataset examples json (batch mode)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--log_path", type=str, help="Path to save log file (JSON format).")
    parser.add_argument("--save_schema", action="store_true", help="Include schema_text in the log file")
    parser.add_argument("--input_mode", choices=["single", "batch"], default="single", help="Control how inputs are provided to the planner",)
    parser.add_argument("--start", type=int, default=0, help="Start index when running in batch mode (0-based)",)
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to run in batch mode (processes all remaining if omitted or <= 0)",)
    args = parser.parse_args()

    tables_meta_path = get_table_paths(args.dataset, args.split, args.tables_path)
    schema_resolver = build_schema_resolver(args.dataset, tables_meta_path)

    graph = PlannerGraph(
        LLMConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    )
    command_line = " ".join(sys.argv)

    if args.log_path:
        log_path = args.log_path
    else:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        default_name = "planner_singlerun_log.json" if args.input_mode == "single" else "planner_batch_log.json"
        log_path = os.path.join(log_dir, default_name)

    if args.input_mode == "single":
        if not args.question or not args.db_id:
            parser.error("question and db_id are required when --input_mode=single")

        schema_text = schema_resolver(args.db_id)
        state = PlannerState(question=args.question, db_id=args.db_id, schema_text=schema_text)
        start_time = time.perf_counter()
        out_dict = graph.invoke(state)
        latency = time.perf_counter() - start_time
        out = PlannerState(**out_dict)

        print("\n=== PLAN ===")
        print(out.plan)
        print("\n=== SQL ===")
        print(out.sql)

        input_args_dict = vars(args).copy()
        save_log(
            log_path=log_path,
            command_line=command_line,
            inputs=input_args_dict,
            plan=out.plan,
            sql=out.sql,
            schema_text=schema_text if args.save_schema else None,
            latency_sec=latency,
        )
        return

    # Batch mode
    examples_file = get_examples_path(args.dataset, args.split, args.examples_path)
    with open(examples_file, "r", encoding="utf-8") as f:
        examples: List[Dict[str, Any]] = json.load(f)

    start = max(0, args.start)
    if start >= len(examples):
        print(f"No examples to process: start index {start} >= total examples {len(examples)}.")
        return

    if args.num_examples is None or args.num_examples <= 0:
        end = len(examples)
    else:
        end = min(len(examples), start + args.num_examples)

    base_inputs = vars(args).copy()
    processed = 0

    for idx in range(start, end):
        example = examples[idx]
        question = example.get("question", "")
        db_id = example.get("db_id")
        if not db_id:
            print(f"[WARN] Example {idx} missing db_id; skipping.")
            continue

        schema_text = schema_resolver(db_id)
        state = PlannerState(question=question, db_id=db_id, schema_text=schema_text)
        try:
            start_time = time.perf_counter()
            out_dict = graph.invoke(state)
            latency = time.perf_counter() - start_time
            out = PlannerState(**out_dict)
        except Exception as exc:
            print(f"[ERROR] Failed to invoke planner for example {idx} ({db_id}): {exc}")
            continue

        print(f"\n=== Example {idx} | db_id={db_id} ===")
        print("Question:", question)
        print("Plan:", out.plan)
        print("SQL:", out.sql)

        log_inputs = base_inputs.copy()
        log_inputs.update({"question": question, "db_id": db_id, "example_index": idx})
        save_log(
            log_path=log_path,
            command_line=command_line,
            inputs=log_inputs,
            plan=out.plan,
            sql=out.sql,
            schema_text=schema_text if args.save_schema else None,
            latency_sec=latency,
        )
        processed += 1

    if processed == 0:
        print("No examples were successfully processed.")
    else:
        print(f"\nProcessed {processed} examples from index {start} to {end - 1}.")


if __name__ == "__main__":
    main()