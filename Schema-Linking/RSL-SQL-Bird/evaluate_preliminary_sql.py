"""
Evaluate execution accuracy of preliminary SQL against gold SQL.
Compares Schema-Linking/RSL-SQL-Bird/src/sql_log/preliminary_sql.txt
with Data/BIRD/dev/dev.sql
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

# Add parent directories to path to import exec_match utilities
# Script is at: Schema-Linking/RSL-SQL-Bird/evaluate_preliminary_sql.py
# Workspace root is: F:\MAster Thesis\CQS-SMART - 1
script_dir = Path(__file__).resolve()
# Go up 3 levels: evaluate_preliminary_sql.py -> RSL-SQL-Bird -> Schema-Linking -> (workspace root)
workspace_root = script_dir.parent.parent.parent
sys.path.insert(0, str(workspace_root))
from baseline.exec_match import (
    exec_sql,
    canonicalize_rows,
    resolve_db_path,
    parse_gold_sql,
    project_root,
)


def read_preliminary_sql(file_path: str) -> List[str]:
    """Read preliminary SQL queries, one per line."""
    with open(file_path, "r", encoding="utf-8") as f:
        queries = []
        for line in f:
            line = line.strip()
            if line:
                queries.append(line.rstrip(";"))
        return queries


def get_db_ids_from_dev_json(dev_json_path: str) -> List[str]:
    """Extract db_id for each example from dev.json."""
    with open(dev_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return [item.get("db_id", "") for item in data]


def evaluate_execution_accuracy(
    preliminary_sql_path: str,
    gold_sql_path: str,
    dev_json_path: str,
    split: str = "dev",
) -> None:
    """Evaluate execution accuracy of preliminary SQL against gold SQL."""
    
    print("Loading files...")
    preliminary_queries = read_preliminary_sql(preliminary_sql_path)
    gold_pairs = parse_gold_sql(gold_sql_path)  # List of (sql, db_id) tuples
    db_ids = get_db_ids_from_dev_json(dev_json_path)
    
    print(f"Found {len(preliminary_queries)} preliminary SQL queries")
    print(f"Found {len(gold_pairs)} gold SQL queries")
    print(f"Found {len(db_ids)} database IDs")
    
    if len(preliminary_queries) != len(gold_pairs):
        print(f"WARNING: Mismatch in query counts: {len(preliminary_queries)} vs {len(gold_pairs)}")
        min_len = min(len(preliminary_queries), len(gold_pairs))
        preliminary_queries = preliminary_queries[:min_len]
        gold_pairs = gold_pairs[:min_len]
        db_ids = db_ids[:min_len]
        print(f"Using first {min_len} queries for evaluation")
    
    # Get database root path
    root = project_root()
    db_root = os.path.join(root, "Data", "BIRD", split, f"{split}_databases")
    
    correct = 0
    total = 0
    errors = []
    
    print("\nEvaluating execution accuracy...")
    print("-" * 80)
    
    for idx, (prelim_sql, (gold_sql, gold_db_id)) in enumerate(zip(preliminary_queries, gold_pairs)):
        # Use db_id from dev.json if available, otherwise use gold_db_id
        db_id = db_ids[idx] if idx < len(db_ids) else gold_db_id
        
        # Resolve database path
        db_path = resolve_db_path(db_root, db_id, dataset="bird", split=split)
        
        if not db_path:
            print(f"[{idx+1}] ERROR: Database not found for db_id='{db_id}'")
            errors.append({
                "index": idx + 1,
                "db_id": db_id,
                "error": "Database not found"
            })
            total += 1
            continue
        
        # Execute both queries
        try:
            prelim_rows = exec_sql(db_path, prelim_sql) if prelim_sql else []
        except Exception as e:
            prelim_rows = []
            errors.append({
                "index": idx + 1,
                "db_id": db_id,
                "error": f"Preliminary SQL execution failed: {e}",
                "sql": prelim_sql[:100] if prelim_sql else ""
            })
        
        try:
            gold_rows = exec_sql(db_path, gold_sql) if gold_sql else []
        except Exception as e:
            gold_rows = []
            errors.append({
                "index": idx + 1,
                "db_id": db_id,
                "error": f"Gold SQL execution failed: {e}",
                "sql": gold_sql[:100] if gold_sql else ""
            })
        
        # Compare results
        prelim_counter = canonicalize_rows(prelim_rows) if prelim_rows else canonicalize_rows([])
        gold_counter = canonicalize_rows(gold_rows) if gold_rows else canonicalize_rows([])
        match = prelim_counter == gold_counter
        
        if match:
            correct += 1
        else:
            errors.append({
                "index": idx + 1,
                "db_id": db_id,
                "error": "Execution results do not match",
                "prelim_sql": prelim_sql[:150] if prelim_sql else "",
                "gold_sql": gold_sql[:150] if gold_sql else "",
                "prelim_rows": len(prelim_rows),
                "gold_rows": len(gold_rows)
            })
        
        total += 1
        
        if (idx + 1) % 50 == 0:
            acc = correct / total if total > 0 else 0.0
            print(f"[{idx+1}/{len(preliminary_queries)}] Current accuracy: {acc:.4f} ({correct}/{total})")
    
    print("-" * 80)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nFinal Execution Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    
    if errors:
        print(f"\nErrors/Non-matches: {len(errors)}")
        # Save errors to file
        error_file = os.path.join(os.path.dirname(preliminary_sql_path), "evaluation_errors.json")
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"Errors saved to: {error_file}")
        
        # Print first 10 errors
        print("\nFirst 10 errors/non-matches:")
        for err in errors[:10]:
            print(f"  [{err['index']}] {err['db_id']}: {err.get('error', 'No match')}")


if __name__ == "__main__":
    root = project_root()
    
    # Use absolute paths based on workspace root
    workspace_root = Path(__file__).parent.parent.parent
    preliminary_sql_path = workspace_root / "Schema-Linking" / "RSL-SQL-Bird" / "src" / "sql_log" / "preliminary_sql.txt"
    # preliminary_sql_path = workspace_root / "Schema-Linking" / "RSL-SQL-Bird" / "src" / "sql_log" / "step_2_information_augmentation.txt"
    gold_sql_path = workspace_root / "Data" / "BIRD" / "dev" / "dev.sql"
    dev_json_path = workspace_root / "Data" / "BIRD" / "dev" / "dev.json"
    
    preliminary_sql_path = str(preliminary_sql_path)
    gold_sql_path = str(gold_sql_path)
    dev_json_path = str(dev_json_path)
    
    if not os.path.exists(preliminary_sql_path):
        print(f"ERROR: Preliminary SQL file not found: {preliminary_sql_path}")
        sys.exit(1)
    
    if not os.path.exists(gold_sql_path):
        print(f"ERROR: Gold SQL file not found: {gold_sql_path}")
        sys.exit(1)
    
    if not os.path.exists(dev_json_path):
        print(f"ERROR: Dev JSON file not found: {dev_json_path}")
        sys.exit(1)
    
    evaluate_execution_accuracy(
        preliminary_sql_path=preliminary_sql_path,
        gold_sql_path=gold_sql_path,
        dev_json_path=dev_json_path,
        split="dev"
    )

