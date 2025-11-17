import os
import sys
import json
import sqlite3
from typing import List, Tuple, Any
import argparse

# Add the current directory to the path to allow imports from same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataloader import load_spider
from planneragent import PlannerGraph
from baseagent import BaseGraph
from state import PlannerState, BaseState
from llm import LLMConfig

# Reuse utilities from run_planner to get schema text and write logs
from run_planner import get_table_paths 
from run_planner import load_schema_text 
from run_planner import save_log 


def project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def ignore_errors_decode(b: bytes) -> str:
    return b.decode(errors='ignore')


def exec_sql(db_path: str, sql: str) -> List[Tuple[Any, ...]]:
    conn = sqlite3.connect(db_path)
    conn.text_factory = ignore_errors_decode
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return rows


def read_examples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_gold_sql(path: str) -> List[Tuple[str, str]]:
    """
    Parse dev_gold.sql lines as-is, preserving duplicates:
      <SQL><TAB><db_id>
    Returns a list of (sql, db_id) tuples.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gold SQL file not found: {path}")

    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            if "\t" in line:
                sql, db_id = line.rsplit("\t", 1)
            else:
                # Fallback for lines without tabs: assume last token is db_id
                parts = line.split()
                if len(parts) < 2:
                    continue
                db_id = parts[-1]
                sql = line[: line.rfind(db_id)].rstrip()
            sql_clean = sql.strip().rstrip(";")
            db_id_clean = db_id.strip()
            if not sql_clean or not db_id_clean:
                continue
            pairs.append((sql_clean, db_id_clean))

    if not pairs:
        raise ValueError(
            f"No gold SQL rows parsed from {path}. "
            "Verify the file format uses tab separators (<SQL>\\t<db_id>)."
        )
    return pairs


def get_spider_paths(split: str) -> tuple[str, str, str, str]:
    root = project_root()
    examples = os.path.join(root, "Data", "spider_data", f"{split}.json")
    tables = os.path.join(
        root, "Data", "spider_data", "test_tables.json" if split == "test" else "tables.json"
    )
    db_root = os.path.join(
        root, "Data", "spider_data", "test_database" if split == "test" else "database"
    )
    gold = os.path.join(root, "Data", "spider_data", f"{split}_gold.sql")
    return examples, tables, db_root, gold


def resolve_db_path(db_root: str, db_id: str, split: str) -> str | None:
    """
    Try multiple common layouts to locate the SQLite file for a db_id.
    Priority:
      1) db_root/db_id/db_id.sqlite
      2) db_root/db_id.sqlite
      3) <project>/Data/spider_data/test_database/db_id/db_id.sqlite
      4) <project>/Data/spider_data/database/db_id/db_id.sqlite
    Returns the first existing path or None.
    """
    candidates: list[str] = []

    # As provided by args
    candidates.append(os.path.join(db_root, db_id, f"{db_id}.sqlite"))
    candidates.append(os.path.join(db_root, f"{db_id}.sqlite"))

    # Standard Spider locations by split
    root = project_root()
    split_pref = ["test_database", "database"] if split == "test" else ["database", "test_database"]
    for folder in split_pref:
        base = os.path.join(root, "Data", "spider_data", folder)
        candidates.append(os.path.join(base, db_id, f"{db_id}.sqlite"))
        candidates.append(os.path.join(base, f"{db_id}.sqlite"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute execution accuracy against Spider gold SQL.")
    parser.add_argument("--split", type=str, default="test", help="Spider split: test or dev")
    parser.add_argument("--examples_path", type=str, help="Override path to Spider examples JSON")
    parser.add_argument("--tables_path", type=str, help="Override path to Spider tables JSON")
    parser.add_argument("--db_root", type=str, help="Override path to Spider DB root folder")
    parser.add_argument("--gold_sql_path", type=str, help="Override path to split_gold.sql")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=10**9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--baseline", choices=["planner", "baseagent"], default="planner", help="Choose which baseline agent to evaluate 'planner' or 'baseagent' ")
    # Logging options consistent with run_planner
    parser.add_argument("--log_path", type=str, help="Path to save log file (JSON). Default: baseline/logs/planner_log.json")
    parser.add_argument("--save_schema", action="store_true", help="Include schema_text in the log file")
    args = parser.parse_args()

    # Resolve paths
    default_examples, default_tables, default_db_root, default_gold = get_spider_paths(args.split)
    examples_path = args.examples_path or default_examples
    db_root = args.db_root or default_db_root
    gold_sql_path = args.gold_sql_path or default_gold

    # Use run_planner's table path resolver to match its behavior
    tables_path = get_table_paths("spider", args.split, args.tables_path)

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
    spider_tables = load_spider(tables_path)  # not strictly needed here, but kept for parity

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

    for idx in range(start, end):
        ex = examples[idx]
        question = ex.get("question", "")
        db_id = ex.get("db_id", "")
        gold_sql, gold_db_id = gold_pairs[idx]

        # Use the example's db_id for schema and db path; gold's db_id is a sanity check
        active_db_id = db_id or gold_db_id

        # Get schema text via run_planner utility (to match planner usage)
        schema_text = load_schema_text("spider", active_db_id, tables_path)

        # Invoke selected agent to generate SQL for the current question
        state = state_cls(question=question, db_id=active_db_id, schema_text=schema_text)
        out_state = None
        try:
            out_dict = graph.invoke(state)
            out_state = state_cls(**out_dict)
            gen_sql = (out_state.sql or "").strip().rstrip(";")
        except Exception as e:
            print(f"[ERROR] Failed to generate plan/SQL for question '{question[:50]}...': {e}")
            gen_sql = ""
            out_state = state_cls(question=question, db_id=active_db_id, schema_text=schema_text)

        # Log using run_planner's logger
        try:
            log_inputs = {
                "question": question,
                "db_id": active_db_id,
                "dataset": "spider",
                "split": args.split,
                "baseline": args.baseline,
                "example_index": idx,
            }
            save_log(
                log_path=log_path,
                command_line=command_line,
                inputs=log_inputs,
                plan=getattr(out_state, "plan", {}) if expects_plan and out_state else {},
                sql=gen_sql,
                schema_text=schema_text if args.save_schema else None
            )
        except Exception as e:
            # Logging shouldn't break evaluation
            print(f"[ERROR] Failed to save log: {e}")
            pass

        # Execute both queries
        db_path = resolve_db_path(db_root, active_db_id, args.split)
        if not db_path:
            print(f"[WARN] SQLite not found for db_id '{active_db_id}'. Checked '{db_root}' and standard Spider locations.")
            total += 1
            continue

        gold_sql_clean = (gold_sql or "").strip().rstrip(";")
        try:
            gen_rows = exec_sql(db_path, gen_sql) if gen_sql else []
        except Exception:
            gen_rows = []
        try:
            gold_rows = exec_sql(db_path, gold_sql_clean) if gold_sql_clean else []
        except Exception:
            gold_rows = []

        total += 1
        # Compare raw execution results exactly as returned by sqlite3.fetchall()
        if gen_rows == gold_rows:
            correct += 1

        if (idx - start + 1) % 25 == 0:
            print(f"[{idx + 1}/{end}] acc={correct/total:.4f}")

    print(f"Execution Accuracy ({total} ex): {correct/total if total else 0.0:.4f}  (correct: {correct})")


if __name__ == "__main__":
    main()