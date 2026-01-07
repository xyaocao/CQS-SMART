import argparse
import collections
import json
from pathlib import Path
from typing import List, Dict, Any


def load_items(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("log", "records", "data", "results"):
            if isinstance(raw.get(key), list):
                return raw[key]
        if all(isinstance(k, str) and k.isdigit() for k in raw.keys()):
            return list(raw.values())
    raise ValueError("Unsupported log structure")


def basic_counts(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    flags = ["exec_match", "execution_match", "execution_correct"]
    counts = {f: sum(1 for x in items if x.get(f) is True) for f in flags}
    return {
        "total": len(items),
        "success": next((v for v in counts.values() if v), 0),
        "flag_counts": counts,
    }


def fail_by_db(items: List[Dict[str, Any]], topn: int = 10) -> List[str]:
    fails = [x for x in items if not x.get("exec_match")]
    counter = collections.Counter(x.get("db_id") for x in fails)
    return [f"{db}: {cnt}" for db, cnt in counter.most_common(topn)]


def chosen_stats(items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    succ = [x for x in items if x.get("exec_match")]
    fail = [x for x in items if not x.get("exec_match")]
    def summarize(arr: List[Dict[str, Any]]) -> List[str]:
        counter = collections.Counter((x.get("reasoner_decision") or {}).get("chosen") for x in arr)
        return [f"{k}: {v}" for k, v in counter.most_common()]
    return {"success": summarize(succ), "fail": summarize(fail)}


def sample_failures(items: List[Dict[str, Any]], db_id: str, limit: int = 3) -> List[Dict[str, str]]:
    fails = [x for x in items if not x.get("exec_match") and x.get("db_id") == db_id][:limit]
    out = []
    for x in fails:
        rd = x.get("reasoner_decision") or {}
        out.append({
            "example_index": str(x.get("example_index")),
            "question": x.get("question") or "",
            "chosen": rd.get("chosen") or "",
            "sql_a": (rd.get("candidate_sql_contract") or "")[:220],
            "sql_b": (rd.get("candidate_sql_oneshot") or "")[:220],
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ImprovedMAD log for mismatch patterns.")
    parser.add_argument("log_path", type=Path, help="Path to ImprovedMAD log JSON")
    parser.add_argument("--topn", type=int, default=10, help="Top N failing dbs to show")
    parser.add_argument("--sample-db", action="append", default=[], help="db_id to show sample failures for (can repeat)")
    args = parser.parse_args()

    items = load_items(args.log_path)
    counts = basic_counts(items)
    print(f"total: {counts['total']}, success: {counts['success']}, flag_counts: {counts['flag_counts']}")

    print("\nTop failing dbs:")
    for line in fail_by_db(items, args.topn):
        print(" -", line)

    print("\nChosen distribution (success/fail):")
    chosen = chosen_stats(items)
    print(" success:", chosen["success"])
    print(" fail   :", chosen["fail"])

    for db_id in args.sample_db:
        print(f"\nSample failures for db {db_id}:")
        for row in sample_failures(items, db_id):
            print(f" example {row['example_index']} | {row['question']}")
            print(f"  chosen: {row['chosen']}")
            print(f"  sql_a: {row['sql_a']}")
            print(f"  sql_b: {row['sql_b']}")


if __name__ == "__main__":
    main()

