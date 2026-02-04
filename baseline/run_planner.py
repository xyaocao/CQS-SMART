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
from baseline.llm import LLMConfig, get_ollama_config
from baseline.exec_match import exec_sql, exec_match

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


def get_db_root(dataset: str, split: str) -> Path:
    """Get the database root directory for a dataset."""
    root = project_root()
    if dataset == "spider":
        return Path(root) / "Data" / "spider_data" / "database"
    elif dataset == "bird":
        return Path(root) / "Data" / "BIRD" / split / "databases"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def save_log(log_path: str, command_line: str, inputs: dict, plan: dict, sql: str, schema_text: str = None,
             latency_sec: float | None = None, is_match: bool | None = None, accuracy_so_far: float | None = None,
             gold_sql: str = None, execution_error: str = None):
    """Save the execution log to a JSON file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "command_line": command_line,
        "inputs": inputs,
        "plan": plan,
        "sql": sql,
    }
    if latency_sec is not None:
        log_entry["latency_sec"] = latency_sec

    if schema_text:
        log_entry["schema_text"] = schema_text

    if is_match is not None:
        log_entry["exec_match"] = is_match

    if accuracy_so_far is not None:
        log_entry["accuracy_so_far"] = accuracy_so_far

    if gold_sql is not None:
        log_entry["gold_sql"] = gold_sql

    if execution_error is not None:
        log_entry["execution_error"] = execution_error
    
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
    # Ollama (local inference)
    parser.add_argument("--use_ollama", action="store_true",
                        help="Use local Ollama instead of SwissAI API")
    parser.add_argument("--ollama_model", type=str, default="ticlazau/qwen2.5-coder-7b-instruct",
                        help="Ollama model name (default: ticlazau/qwen2.5-coder-7b-instruct)")
    args = parser.parse_args()

    tables_meta_path = get_table_paths(args.dataset, args.split, args.tables_path)
    schema_resolver = build_schema_resolver(args.dataset, tables_meta_path)

    # Setup LLM config
    if args.use_ollama:
        llm_config = get_ollama_config(
            model=args.ollama_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"Using Ollama with model: {args.ollama_model}")
    else:
        llm_config = LLMConfig(temperature=args.temperature, max_tokens=args.max_tokens)

    graph = PlannerGraph(llm_config)
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

    # Get database root for execution
    db_root = get_db_root(args.dataset, args.split)

    base_inputs = vars(args).copy()
    processed = 0
    correct = 0
    total = 0

    print(f"\nRunning batch mode: {args.dataset}/{args.split}")
    print(f"Examples: {start} to {end - 1}")
    print(f"Log path: {log_path}")
    print("-" * 50)

    for idx in range(start, end):
        example = examples[idx]
        question = example.get("question", "")
        db_id = example.get("db_id")
        gold_sql = example.get("query", example.get("sql", ""))
        if not db_id:
            print(f"[WARN] Example {idx} missing db_id; skipping.")
            continue

        schema_text = schema_resolver(db_id)
        state = PlannerState(question=question, db_id=db_id, schema_text=schema_text)

        execution_error = None
        is_match = False
        plan = {}
        final_sql = ""

        try:
            start_time = time.perf_counter()
            out_dict = graph.invoke(state)
            latency = time.perf_counter() - start_time
            out = PlannerState(**out_dict)
            plan = out.plan
            final_sql = out.sql
        except Exception as exc:
            print(f"[ERROR] Failed to invoke planner for example {idx} ({db_id}): {exc}")
            latency = time.perf_counter() - start_time if 'start_time' in locals() else 0
            execution_error = str(exc)

        # Evaluate execution match
        db_path = db_root / db_id / f"{db_id}.sqlite"
        try:
            gen_rows = exec_sql(str(db_path), final_sql)
            gold_rows = exec_sql(str(db_path), gold_sql)
            is_match, _ = exec_match(gen_rows, gold_rows, order_matters=False, match_mode="hard")
        except Exception as e:
            is_match = False
            execution_error = str(e)

        if is_match:
            correct += 1
        total += 1
        accuracy = correct / total if total > 0 else 0.0

        # Progress output
        status = "+" if is_match else "X"
        print(f"[{idx:4d}] {status} Acc: {accuracy:.2%} | {question[:60]}...")

        log_inputs = base_inputs.copy()
        log_inputs.update({"question": question, "db_id": db_id, "example_index": idx})
        save_log(
            log_path=log_path,
            command_line=command_line,
            inputs=log_inputs,
            plan=plan,
            sql=final_sql,
            schema_text=schema_text if args.save_schema else None,
            latency_sec=latency,
            is_match=is_match,
            accuracy_so_far=accuracy,
            gold_sql=gold_sql,
            execution_error=execution_error
        )
        processed += 1

    # Final summary
    print("\n" + "=" * 50)
    print(f"Final Results:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/total:.2%}" if total > 0 else "  Accuracy: N/A")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()