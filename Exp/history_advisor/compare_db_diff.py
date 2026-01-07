"""Compare exec_match differences for a single db between two logs and optionally
emit improved indices for history advisor updates.

Usage:
  python Exp/history_advisor/compare_db_diff.py \
    --db orchestra \
    --old Exp/logs/Qwen2.5-Coder/ImprovedMad_run1_log.json \
    --new Exp/logs/Qwen2.5-Coder/test/improvedmad_new_orchestra.json
"""

import argparse
import json
import pathlib
from typing import Any, Dict, List


def load_items(path: pathlib.Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("log", "records", "data", "results"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise ValueError(f"Unsupported log structure in {path}")


def index_exec_map(items: List[Dict[str, Any]], db_id: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for x in items:
        if x.get("db_id") != db_id:
            continue
        idx = x.get("example_index")
        if idx is None:
            continue
        out[int(idx)] = x
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff exec_match for a DB between two logs.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--old", required=True, type=pathlib.Path)
    parser.add_argument("--new", required=True, type=pathlib.Path)
    args = parser.parse_args()

    old_items = load_items(args.old)
    new_items = load_items(args.new)

    old_map = index_exec_map(old_items, args.db)
    new_map = index_exec_map(new_items, args.db)

    improved = []
    regressed = []
    unchanged = []

    for idx, old_row in old_map.items():
        new_row = new_map.get(idx)
        if not new_row:
            continue
        old_val = bool(old_row.get("exec_match"))
        new_val = bool(new_row.get("exec_match"))
        if old_val and not new_val:
            regressed.append(idx)
        elif not old_val and new_val:
            improved.append(idx)
        else:
            unchanged.append(idx)

    print(f"DB: {args.db}")
    print(f"Total compared: {len(improved)+len(regressed)+len(unchanged)}")
    print(f"Improved (false -> true): {len(improved)}")
    print(f"Regressed (true -> false): {len(regressed)}")
    if improved:
        print("Improved indices:", sorted(improved))
        # Show question + final_sql for each improved
        for idx in sorted(improved):
            row = new_map[idx]
            # print the question and final SQL for this improved example
            print(f"\n[idx {idx}] question: {row.get('question')}")
            sql = (row.get("final_sql") or "").strip()
            print(f"final_sql:\n{sql}")

    if regressed:
        print("Regressed indices:", sorted(regressed))


if __name__ == "__main__":
    main()