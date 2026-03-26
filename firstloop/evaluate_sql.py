"""
Evaluate execution accuracy of SQL queries against gold SQL.
Automatically calculates Hard, Soft, and Partial Execution Accuracy.
Similar to RSL-SQL-Spider evaluate_preliminary_sql.py but with CLI arguments.

Usage:
    python Exp/evaluate_sql.py \
        --evaluate_sql_file Exp/sql_log/refined_sql.txt \
        --gold_sql_path Data/spider_data/dev_gold.sql \
        --dev_json_path Data/spider_data/dev.json \
        --db_root Data/spider_data/database \
        --split dev \
        --output_errors Exp/sql_log/evaluation_errors.json

    python Exp/evaluate_sql.py --evaluate_sql_file results/extracted_sql/BIRD/run1_log.txt --gold_sql_path Data/BIRD/dev/dev.sql   --dev_json_path Data/BIRD/dev/dev.json --db_root Data/BIRD/dev/dev_databases --split dev 
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

# Add baseline to path
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))

from baseline.exec_match import (
    exec_sql,
    canonicalize_rows,
    exec_match,
    resolve_db_path,
    parse_gold_sql,
    project_root,
)


def read_sql_file(file_path: str) -> List[str]:
    """Read SQL queries from file, one per line."""
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
    evaluate_sql_path: str,
    gold_sql_path: str,
    dev_json_path: str = None,
    db_root: str = None,
    split: str = "dev",
    output_errors: str = None,
) -> None:
    """
    Evaluate execution accuracy of SQL queries against gold SQL.
    
    Args:
        evaluate_sql_path: Path to SQL file to evaluate (one SQL per line)
        gold_sql_path: Path to gold SQL file (format: SQL<TAB>db_id)
        dev_json_path: Optional path to dev.json for db_id mapping
        db_root: Optional database root directory
        split: Dataset split (dev/test)
        output_errors: Optional path to save error details
    """
    print("=" * 80)
    print("SQL Execution Accuracy Evaluation (Hard, Soft, Partial)")
    print("=" * 80)
    print(f"Evaluating SQL file: {evaluate_sql_path}")
    print(f"Gold SQL file: {gold_sql_path}")
    print()
    
    # Load files
    print("Loading files...")
    evaluate_queries = read_sql_file(evaluate_sql_path)
    gold_pairs = parse_gold_sql(gold_sql_path)  # List of (sql, db_id) tuples
    
    print(f"Found {len(evaluate_queries)} SQL queries to evaluate")
    print(f"Found {len(gold_pairs)} gold SQL queries")
    
    # Get db_ids if dev.json is provided
    db_ids = None
    if dev_json_path and os.path.exists(dev_json_path):
        db_ids = get_db_ids_from_dev_json(dev_json_path)
        print(f"Found {len(db_ids)} database IDs from dev.json")
    else:
        print("No dev.json provided, using db_ids from gold SQL file")
    
    # Check length mismatch
    if len(evaluate_queries) != len(gold_pairs):
        print(f"\nWARNING: Mismatch in query counts: {len(evaluate_queries)} vs {len(gold_pairs)}")
        min_len = min(len(evaluate_queries), len(gold_pairs))
        evaluate_queries = evaluate_queries[:min_len]
        gold_pairs = gold_pairs[:min_len]
        if db_ids:
            db_ids = db_ids[:min_len]
        print(f"Using first {min_len} queries for evaluation")
    
    # Get database root path
    if not db_root:
        root = project_root()
        db_root = os.path.join(root, "Data", "spider_data", "database")
    
    if not os.path.exists(db_root):
        print(f"WARNING: Database root not found: {db_root}")
        print("Trying alternative paths...")
        # Try alternative paths
        root = project_root()
        alt_paths = [
            os.path.join(root, "Data", "spider_data", "database"),
            os.path.join(root, "database"),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                db_root = alt_path
                print(f"Using database root: {db_root}")
                break
        else:
            print(f"ERROR: Could not find database root. Please specify --db_root")
            sys.exit(1)
    
    print(f"Database root: {db_root}")
    print()
    
    # Evaluation
    hard_correct = 0
    soft_correct = 0
    partial_correct = 0
    total = 0
    errors = []
    detailed_results = []
    
    print("Evaluating execution accuracy (Hard, Soft, Partial)...")
    print("-" * 80)
    
    for idx, (eval_sql, (gold_sql, gold_db_id)) in enumerate(zip(evaluate_queries, gold_pairs)):
        # Use db_id from dev.json if available, otherwise use gold_db_id
        db_id = db_ids[idx] if db_ids and idx < len(db_ids) else gold_db_id
        
        # Resolve database path
        db_path = resolve_db_path(db_root, db_id, split=split, dataset="spider")
        
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
            eval_rows = exec_sql(db_path, eval_sql) if eval_sql else []
        except Exception as e:
            eval_rows = []
            errors.append({
                "index": idx + 1,
                "db_id": db_id,
                "error": f"Evaluated SQL execution failed: {e}",
                "sql": eval_sql[:100] if eval_sql else ""
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
        
        # Compare results using all three modes: Hard, Soft, Partial
        # Hard match: exact match with column/row permutation (using canonicalize_rows for backward compatibility)
        eval_counter = canonicalize_rows(eval_rows) if eval_rows else canonicalize_rows([])
        gold_counter = canonicalize_rows(gold_rows) if gold_rows else canonicalize_rows([])
        hard_match = eval_counter == gold_counter
        
        # Soft and Partial matches: use exec_match function
        soft_match, soft_details = exec_match(gold_rows, eval_rows, order_matters=False, match_mode="soft")
        partial_match, partial_details = exec_match(gold_rows, eval_rows, order_matters=False, match_mode="partial")
        
        if hard_match:
            hard_correct += 1
        if soft_match:
            soft_correct += 1
        if partial_match:
            partial_correct += 1
        
        # Track detailed results
        detailed_results.append({
            "index": idx + 1,
            "db_id": db_id,
            "hard_match": hard_match,
            "soft_match": soft_match,
            "partial_match": partial_match,
            "eval_rows_count": len(eval_rows),
            "gold_rows_count": len(gold_rows),
        })
        
        # Add to errors if none of the modes match
        if not hard_match and not soft_match and not partial_match:
            errors.append({
                "index": idx + 1,
                "db_id": db_id,
                "error": "Execution results do not match (hard, soft, or partial)",
                "eval_sql": eval_sql[:150] if eval_sql else "",
                "gold_sql": gold_sql[:150] if gold_sql else "",
                "eval_rows": len(eval_rows),
                "gold_rows": len(gold_rows),
                "hard_match": hard_match,
                "soft_match": soft_match,
                "partial_match": partial_match
            })
        
        total += 1
        
        # Progress update
        if (idx + 1) % 50 == 0:
            hard_acc = hard_correct / total if total > 0 else 0.0
            soft_acc = soft_correct / total if total > 0 else 0.0
            partial_acc = partial_correct / total if total > 0 else 0.0
            print(f"[{idx+1}/{len(evaluate_queries)}] Hard: {hard_acc:.4f}, Soft: {soft_acc:.4f}, Partial: {partial_acc:.4f}")
    
    # Final results
    print("-" * 80)
    hard_accuracy = hard_correct / total if total > 0 else 0.0
    soft_accuracy = soft_correct / total if total > 0 else 0.0
    partial_accuracy = partial_correct / total if total > 0 else 0.0
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Hard Execution Accuracy: {hard_correct}/{total} = {hard_accuracy:.4f} ({hard_accuracy*100:.2f}%)")
    print(f"Soft Execution Accuracy: {soft_correct}/{total} = {soft_accuracy:.4f} ({soft_accuracy*100:.2f}%)")
    print(f"Partial Execution Accuracy: {partial_correct}/{total} = {partial_accuracy:.4f} ({partial_accuracy*100:.2f}%)")
    print(f"\nTotal examples: {total}")
    print(f"Hard correct: {hard_correct}")
    print(f"Soft correct: {soft_correct}")
    print(f"Partial correct: {partial_correct}")
    print(f"Incorrect/Errors (all modes failed): {total - partial_correct}")
    
    # Save detailed results
    eval_dir = os.path.dirname(evaluate_sql_path)
    detailed_results_path = Path(eval_dir) / "evaluation_detailed_results.json"
    detailed_results_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "total_examples": total,
        "hard_execution_accuracy": hard_accuracy,
        "soft_execution_accuracy": soft_accuracy,
        "partial_execution_accuracy": partial_accuracy,
        "hard_correct": hard_correct,
        "soft_correct": soft_correct,
        "partial_correct": partial_correct,
        "detailed_results": detailed_results
    }
    
    with open(detailed_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {detailed_results_path}")
    
    if errors:
        print(f"\nErrors/Non-matches (all modes failed): {len(errors)}")
        
        # Save errors to file
        if output_errors:
            error_file = output_errors
        else:
            # Default: save in same directory as evaluate_sql_path
            error_file = os.path.join(eval_dir, "evaluation_errors.json")
        
        error_path = Path(error_file)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"Errors saved to: {error_file}")
        
        # Print first 10 errors
        print("\nFirst 10 errors/non-matches:")
        for err in errors[:10]:
            match_info = []
            if err.get('hard_match'):
                match_info.append("hard")
            if err.get('soft_match'):
                match_info.append("soft")
            if err.get('partial_match'):
                match_info.append("partial")
            match_str = f" (passed: {', '.join(match_info)})" if match_info else ""
            print(f"  [{err['index']}] {err['db_id']}: {err.get('error', 'No match')}{match_str}")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate execution accuracy of SQL queries against gold SQL. Automatically calculates Hard, Soft, and Partial Execution Accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (automatically calculates Hard, Soft, Partial EX)
  python Exp/evaluate_sql.py \\
      --evaluate_sql_file Exp/sql_log/refined_sql.txt \\
      --gold_sql_path Data/spider_data/dev_gold.sql
  
  # With dev.json for db_id mapping
  python Exp/evaluate_sql.py \\
      --evaluate_sql_file Exp/sql_log/refined_sql.txt \\
      --gold_sql_path Data/spider_data/dev_gold.sql \\
      --dev_json_path Data/spider_data/dev.json
  
  # With custom database root
  python Exp/evaluate_sql.py \\
      --evaluate_sql_file Exp/sql_log/refined_sql.txt \\
      --gold_sql_path Data/spider_data/dev_gold.sql \\
      --db_root Data/spider_data/database

Note: This script automatically evaluates all three modes:
  - Hard EX: Exact match with column/row permutation
  - Soft EX: Generated SQL must contain all columns from gold SQL
  - Partial EX: Generated SQL is a subset of gold SQL
        """
    )
    
    parser.add_argument(
        "--evaluate_sql_file",
        type=str,
        required=True,
        help="Path to SQL file to evaluate (one SQL per line)"
    )
    parser.add_argument(
        "--gold_sql_path",
        type=str,
        required=True,
        help="Path to gold SQL file (format: SQL<TAB>db_id)"
    )
    parser.add_argument(
        "--dev_json_path",
        type=str,
        default=None,
        help="Optional path to dev.json for db_id mapping"
    )
    parser.add_argument(
        "--db_root",
        type=str,
        default=None,
        help="Optional database root directory (default: Data/spider_data/database)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Dataset split (default: dev)"
    )
    parser.add_argument(
        "--output_errors",
        type=str,
        default=None,
        help="Optional path to save error details JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.evaluate_sql_file):
        print(f"ERROR: SQL file not found: {args.evaluate_sql_file}")
        sys.exit(1)
    
    if not os.path.exists(args.gold_sql_path):
        print(f"ERROR: Gold SQL file not found: {args.gold_sql_path}")
        sys.exit(1)
    
    if args.dev_json_path and not os.path.exists(args.dev_json_path):
        print(f"WARNING: Dev JSON file not found: {args.dev_json_path}")
        print("Continuing without db_id mapping from dev.json...")
        args.dev_json_path = None
    
    # Run evaluation
    evaluate_execution_accuracy(
        evaluate_sql_path=args.evaluate_sql_file,
        gold_sql_path=args.gold_sql_path,
        dev_json_path=args.dev_json_path,
        db_root=args.db_root,
        split=args.split,
        output_errors=args.output_errors,
    )

