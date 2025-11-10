import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add the current directory to the path to allow imports from same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataloader import load_spider, get_schema_from_spider, load_bird, get_schema_from_bird
from planneragent import PlannerGraph
from state import PlannerState
from swissai import SwissAIConfig

def project_root() -> str:
    """Get the root directory of the project."""
    # baseline is now at root level, so go up one directory from baseline
    return os.path.dirname(os.path.dirname(__file__))

def get_table_paths(dataset: str, split: str, tables_path: str | None) -> str:
    if tables_path:
        return tables_path
    root = project_root()
    if dataset == "spider":
        default_file = "test_tables.json" if split == "test" else "tables.json"
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


def save_log(log_path: str, question: str, db_id: str, dataset: str, split: str, 
             plan: dict, sql: str, schema_text: str = None):
    """Save the execution log to a JSON file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "db_id": db_id,
        "dataset": dataset,
        "split": split,
        "plan": plan,
        "sql": sql,
    }
    
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
    
    print(f"\n=== Log saved to: {log_path} ===")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="Natural language question")
    parser.add_argument("db_id", type=str, help="Database id (folder name)")
    parser.add_argument("--dataset", choices=["bird", "spider"], default="spider")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--tables_path", type=str, help="Override path to tables metadata json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--log_path", type=str, help="Path to save log file (JSON format). Default: baseline/logs/planner_log.json")
    parser.add_argument("--save_schema", action="store_true", help="Include schema_text in the log file")
    args = parser.parse_args()

    tables_meta_path = get_table_paths(args.dataset, args.split, args.tables_path)
    schema_text = load_schema_text(args.dataset, args.db_id, tables_meta_path)

    graph = PlannerGraph(
        SwissAIConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    )
    state = PlannerState(question=args.question, db_id=args.db_id, schema_text=schema_text)
    out_dict = graph.invoke(state)
    # Convert dict back to PlannerState for easier access
    out = PlannerState(**out_dict)

    print("\n=== PLAN ===")
    print(out.plan)
    print("\n=== SQL ===")
    print(out.sql)
    
    # Save to log file
    if args.log_path:
        log_path = args.log_path
    else:
        # Default log path: baseline/logs/planner_log.json
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        log_path = os.path.join(log_dir, "planner_log.json")
    
    save_log(
        log_path=log_path,
        question=args.question,
        db_id=args.db_id,
        dataset=args.dataset,
        split=args.split,
        plan=out.plan,
        sql=out.sql,
        schema_text=schema_text if args.save_schema else None
    )


if __name__ == "__main__":
    main()