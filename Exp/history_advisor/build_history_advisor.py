"""Build a high-signal history_advisor.jsonl from a run log.

Rules:
 - Use only exec_match == True rows (verified successes).
 - Keep only exemplars that demonstrate disambiguation value:
     join_fk, min_max (ORDER BY + LIMIT / min/max), group_by, distinct, string_match,
     or count combined with group_by/distinct.
 - Skip trivial count-only / bare select cases.
 - Per-db cap to maintain balance.

Usage:
  python Exp/history_advisor/build_history_advisor.py \
      --log Exp/logs/Qwen3/ImprovedMad_run1_log.json \
      --out Exp/history_advisor.jsonl \
      --per-db-cap 8
"""

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Iterable, Dict, Any, List, Tuple


def load_items(path: pathlib.Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("log", "records", "data", "results"):
            if isinstance(raw.get(key), list):
                return raw[key]
    raise ValueError("Unsupported log structure")


def tag_sql(sql: str) -> Tuple[List[str], List[str]]:
    s = (sql or "").lower()
    tags = set()
    rationale: List[str] = []
    if " join " in s:
        tags.add("join_fk")
        rationale.append("Uses joins to link required tables")
    if "count(" in s:
        tags.add("count")
        rationale.append("Uses count aggregation")
    if "distinct" in s:
        tags.add("distinct")
        rationale.append("Uses distinct to dedup results")
    if "group by" in s:
        tags.add("group_by")
        rationale.append("Groups non-aggregated columns")
    if ("order by" in s and "limit" in s) or " min(" in s or " max(" in s:
        tags.add("min_max")
        rationale.append("Orders by metric with limit for extremum")
    if " like " in s or "lower(" in s:
        tags.add("string_match")
        rationale.append("Handles string match with LIKE/LOWER")
    if not rationale:
        rationale.append("Direct selection per schema columns")
    return sorted(tags), rationale[:3]


def is_high_signal(tags: List[str]) -> bool:
    sig = {"join_fk", "min_max", "group_by", "distinct", "string_match"}
    if sig.intersection(tags):
        return True
    # allow count only when paired with group_by or distinct
    if "count" in tags and ("group_by" in tags or "distinct" in tags):
        return True
    return False


def build_advisor(items: Iterable[Dict[str, Any]], targets: List[str], per_db_cap: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    used = defaultdict(int)
    # sort by db, then example_index for determinism
    sorted_items = sorted(items, key=lambda y: (str(y.get("db_id") or ""), y.get("example_index", 0)))
    for x in sorted_items:
        if not x.get("exec_match"):
            continue
        db = (x.get("db_id") or "").strip()
        if db not in targets:
            continue
        if used[db] >= per_db_cap:
            continue
        sql = (x.get("final_sql") or "").strip()
        if not sql:
            continue
        tags, rationale = tag_sql(sql)
        if not is_high_signal(tags):
            continue
        q = (x.get("question") or "").strip()
        out.append(
            {
                "db_id": db,
                "question": q,
                "answer_sql": sql,
                "rationale": rationale,
                "tags": tags,
                "source": {
                    "example_index": x.get("example_index"),
                    "run": "ImprovedMad_run1_Qwen3",
                    "chosen": "final_sql",
                },
            }
        )
        used[db] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-signal history advisor examples from a log.")
    parser.add_argument("--log", required=True, type=pathlib.Path, help="Path to run log JSON")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="Output JSONL path")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "car_1",
            "cre_Doc_Template_Mgt",
            "flight_2",
            "wta_1",
            "student_transcripts_tracking",
            "dog_kennels",
            "tvshow",
            "network_1",
            "orchestra",
            "world_1",
        ],
        help="DB IDs to extract exemplars for",
    )
    parser.add_argument("--per-db-cap", type=int, default=8, help="Max exemplars per db")
    args = parser.parse_args()

    items = load_items(args.log)
    advisor = build_advisor(items, args.targets, args.per_db_cap)
    args.out.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in advisor), encoding="utf-8")
    print(f"wrote {len(advisor)} entries to {args.out}")


if __name__ == "__main__":
    main()

