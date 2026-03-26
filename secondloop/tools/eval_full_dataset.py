"""
eval_full_dataset.py

Calculate execution accuracy (hard / soft / partial) and per-stage latency
for one or more log files against the *whole* BIRD or Spider dataset.

Accuracy is re-computed by actually executing both the generated SQL and the
gold SQL against the database, then comparing result sets with the three
matching modes from baseline/exec_match.py.

Latency is read from the recorded latency fields in each log entry — the same
stage keys used in compare_ablation.py — so no re-timing occurs.

Usage
-----
    # Single log, BIRD dev
    python secondloop/tools/eval_full_dataset.py --dataset bird --split dev \\
        secondloop/logs/bird/qwen3/online/run1.json

    # Spider dev, compare two configs side by side
    python secondloop/tools/eval_full_dataset.py --dataset spider --split dev \\
        secondloop/logs/spider/offline/run_qwen3.json \\
        secondloop/logs/spider/offline/run_other.json

    # Restrict to a saved index subset (e.g. the 500-example bird subset)
    python secondloop/tools/eval_full_dataset.py --dataset bird --split dev \\
        --index_file secondloop/tools/bird_dev_500.json \\
        secondloop/logs/bird/qwen3/online/run1.json

    # Save the structured report to a custom path
    python secondloop/tools/eval_full_dataset.py --dataset bird --split dev \\
        --output secondloop/logs/bird/ablation/full_eval_report.json \\
        secondloop/logs/bird/qwen3/online/run1.json
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Make the project root importable
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from baseline.exec_match import (
    exec_match,
    exec_sql,
    get_dataset_paths,
    parse_gold_sql,
    resolve_db_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MATCH_MODES = ["hard", "soft", "partial"]

STAGE_KEYS = [
    "planner_sec",
    "sqlgen_sec",
    "reviewer_sec",
    "regeneration_sec",
    "regeneration_retry_sec",
    "verification_sec",
    "verification_regen_sec",
    "refiner_sec",
    "schema_linking_sec",
    "total_sec",
    "pipeline_total_sec",
]

BIRD_DIFFICULTIES = ["simple", "moderate", "challenging"]
SPIDER_DIFFICULTIES = ["easy", "medium", "hard", "extra"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d > 0 else "N/A"


def fmt_stat(values: List[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    med = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 2 else mean
    return f"mean={mean:.1f}s  med={med:.1f}s  p95={p95:.1f}s"


def load_bird_difficulty_map(split: str) -> Dict[int, str]:
    """Index → difficulty for BIRD (from dev.json / train.json)."""
    examples_path = BASE_DIR / "Data" / "BIRD" / split / f"{split}.json"
    if not examples_path.exists():
        return {}
    with open(examples_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {i: item.get("difficulty", "unknown").lower() for i, item in enumerate(data)}


def load_spider_difficulty_map(split: str) -> Dict[int, str]:
    """Spider dev/test examples have no difficulty label — return empty map."""
    return {}


def load_gold_sql_map(dataset: str, split: str) -> Dict[int, Tuple[str, str]]:
    """
    Return {index: (sql, db_id)} for every gold example.
    Reads from the .sql gold file for the given dataset/split.
    """
    _, _, _, gold_path = get_dataset_paths(dataset, split)
    pairs = parse_gold_sql(gold_path)
    return {i: pair for i, pair in enumerate(pairs)}


def get_generated_sql(record: Dict) -> Optional[str]:
    """
    Extract the final generated SQL from a log record.
    Preference order:
      1. sql_tracking.final_sql
      2. latency.gen_sql_executed
      3. sql_tracking.revised_sql
      4. sql_tracking.initial_sql
    """
    st = record.get("sql_tracking", {})
    for key in ("final_sql", "revised_sql", "initial_sql"):
        val = st.get(key)
        if val and val.strip():
            return val.strip().rstrip(";")
    lat = record.get("latency", {})
    val = lat.get("gen_sql_executed")
    if val and val.strip():
        return val.strip().rstrip(";")
    return None


def safe_exec(db_path: str, sql: str, timeout: float = 30.0):
    """Execute SQL, returning (rows, error_str)."""
    try:
        rows = exec_sql(db_path, sql, timeout=timeout)
        return rows, None
    except TimeoutError as e:
        return None, f"TIMEOUT: {e}"
    except Exception as e:
        return None, f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _eval_one(
    idx: int,
    record: Dict,
    diff: str,
    gold_map: Dict[int, Tuple[str, str]],
    db_root: str,
    dataset: str,
    split: str,
    sql_timeout: float,
) -> Dict[str, Any]:
    """
    Evaluate a single example. Designed to run inside a thread pool.
    Returns a plain dict; no shared mutable state is touched here.
    """
    # --- Latency & rates (cheap, no I/O) ---
    lat = record.get("latency", {})
    lat_values = {
        key: float(lat[key])
        for key in STAGE_KEYS
        if lat.get(key) is not None and isinstance(lat.get(key), (int, float)) and lat[key] > 0
    }

    st = record.get("sql_tracking", {})
    dbg = record.get("debug", {})
    rates = {
        "revision": bool(st.get("revision_applied")),
        "refiner": bool(st.get("refiner_applied")),
        "verification": bool(st.get("verification_applied")),
        "verify_regen": bool(dbg and dbg.get("verification_triggered_regen")),
    }

    # --- SQL execution ---
    gen_sql = get_generated_sql(record)
    if not gen_sql:
        return {"idx": idx, "diff": diff, "skip": "no_sql", "lat": lat_values, "rates": rates}

    gold_entry = gold_map.get(idx)
    if gold_entry is None:
        return {"idx": idx, "diff": diff, "skip": "no_gold", "lat": lat_values, "rates": rates}
    gold_sql, db_id = gold_entry

    db_path = resolve_db_path(db_root, db_id, split, dataset)
    if db_path is None:
        return {"idx": idx, "diff": diff, "skip": "no_db", "lat": lat_values, "rates": rates}

    gen_rows, gen_err = safe_exec(db_path, gen_sql, timeout=sql_timeout)
    gold_rows, gold_err = safe_exec(db_path, gold_sql, timeout=sql_timeout)

    if gold_err:
        return {"idx": idx, "diff": diff, "skip": "gold_err", "lat": lat_values, "rates": rates}
    if gen_err:
        return {"idx": idx, "diff": diff, "skip": "gen_err", "lat": lat_values, "rates": rates}

    matches = {
        mode: exec_match(gen_rows, gold_rows, order_matters=False, match_mode=mode)[0]
        for mode in MATCH_MODES
    }
    return {"idx": idx, "diff": diff, "skip": None, "matches": matches, "lat": lat_values, "rates": rates}


def analyse_log(
    log_path: Path,
    dataset: str,
    split: str,
    gold_map: Dict[int, Tuple[str, str]],
    diff_map: Dict[int, str],
    subset_indices: Optional[Set[int]],
    sql_timeout: float,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Analyse a single log file using a thread pool for parallel SQL execution.

    Returns a rich dict with accuracy (per match_mode, per difficulty),
    latency stats, and pipeline rates.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = [records]

    # Index by example_index
    by_idx: Dict[int, Dict] = {}
    for r in records:
        idx = r.get("example_index")
        if idx is not None:
            by_idx[idx] = r

    # Determine which indices to evaluate
    if subset_indices is not None:
        indices = sorted(i for i in subset_indices if i in by_idx)
        missing = sorted(i for i in subset_indices if i not in by_idx)
    else:
        indices = sorted(by_idx.keys())
        missing = []

    _, _, db_root, _ = get_dataset_paths(dataset, split)

    # --- Parallel evaluation ---
    print(f"  Running {len(indices)} examples with {workers} workers ...", flush=True)
    per_example_results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _eval_one,
                idx,
                by_idx[idx],
                diff_map.get(idx, "unknown"),
                gold_map,
                db_root,
                dataset,
                split,
                sql_timeout,
            ): idx
            for idx in indices
        }
        done = 0
        for future in as_completed(futures):
            per_example_results.append(future.result())
            done += 1
            if done % 100 == 0 or done == len(indices):
                print(f"  ... {done}/{len(indices)}", flush=True)

    # --- Aggregate ---
    accuracy: Dict[str, Dict[str, int]] = {m: {"correct": 0, "total": 0} for m in MATCH_MODES}
    diff_accuracy: Dict[str, Dict[str, Dict[str, int]]] = {
        m: defaultdict(lambda: {"correct": 0, "total": 0}) for m in MATCH_MODES
    }
    stage_latencies: Dict[str, List[float]] = defaultdict(list)
    revision_count = refiner_count = verification_count = verification_triggered_regen = 0
    exec_errors = gold_errors = skipped_no_sql = skipped_no_gold = skipped_no_db = 0

    for er in per_example_results:
        diff = er["diff"]

        # Latency (always collected regardless of skip)
        for key, val in er["lat"].items():
            stage_latencies[key].append(val)

        # Rates
        r = er["rates"]
        if r["revision"]:
            revision_count += 1
        if r["refiner"]:
            refiner_count += 1
        if r["verification"]:
            verification_count += 1
        if r["verify_regen"]:
            verification_triggered_regen += 1

        # Skip accounting
        skip = er.get("skip")
        if skip == "no_sql":
            skipped_no_sql += 1
            continue
        if skip == "no_gold":
            skipped_no_gold += 1
            continue
        if skip == "no_db":
            skipped_no_db += 1
            continue
        if skip == "gold_err":
            gold_errors += 1
            continue
        if skip == "gen_err":
            exec_errors += 1
            for mode in MATCH_MODES:
                accuracy[mode]["total"] += 1
                diff_accuracy[mode][diff]["total"] += 1
            continue

        for mode in MATCH_MODES:
            is_match = er["matches"][mode]
            accuracy[mode]["total"] += 1
            diff_accuracy[mode][diff]["total"] += 1
            if is_match:
                accuracy[mode]["correct"] += 1
                diff_accuracy[mode][diff]["correct"] += 1

    return {
        "log_path": str(log_path),
        "log_name": log_path.stem,
        "dataset": dataset,
        "split": split,
        "total_in_log": len(records),
        "indices_evaluated": len(indices),
        "subset_missing": len(missing),
        "skipped_no_sql": skipped_no_sql,
        "skipped_no_gold": skipped_no_gold,
        "skipped_no_db": skipped_no_db,
        "exec_errors": exec_errors,
        "gold_errors": gold_errors,
        "accuracy": {m: dict(v) for m, v in accuracy.items()},
        "diff_accuracy": {
            m: {d: dict(v) for d, v in diffs.items()}
            for m, diffs in diff_accuracy.items()
        },
        "stage_latencies": dict(stage_latencies),
        "revision_count": revision_count,
        "refiner_count": refiner_count,
        "verification_count": verification_count,
        "verification_triggered_regen": verification_triggered_regen,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _difficulty_order(dataset: str) -> List[str]:
    if dataset.lower() == "bird":
        return BIRD_DIFFICULTIES
    return SPIDER_DIFFICULTIES


def print_report(result: Dict):
    dataset = result["dataset"]
    diffs = _difficulty_order(dataset)
    n_eval = result["indices_evaluated"]

    print(f"\n{'=' * 70}")
    print(f"  Log   : {result['log_name']}")
    print(f"  Path  : {result['log_path']}")
    print(f"  Data  : {dataset.upper()} / {result['split']}")
    print(f"{'=' * 70}")

    print(f"\n  Records in log : {result['total_in_log']}")
    print(f"  Evaluated      : {n_eval}")
    if result["subset_missing"]:
        print(f"  WARNING: {result['subset_missing']} subset indices not in log")
    for key in ("skipped_no_sql", "skipped_no_gold", "skipped_no_db", "exec_errors", "gold_errors"):
        val = result.get(key, 0)
        if val:
            print(f"  WARNING: {key} = {val}")

    # --- Accuracy table ---
    print(f"\n  Execution Accuracy")
    print(f"  {'Mode':<10s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"  {'-' * 40}")
    for mode in MATCH_MODES:
        acc = result["accuracy"][mode]
        c, t = acc["correct"], acc["total"]
        print(f"  {mode:<10s} {c:>8d} {t:>8d} {pct(c, t):>10s}")

    # --- Per-difficulty (only when difficulty info is available) ---
    hard_diff = result["diff_accuracy"].get("hard", {})
    known_diffs = [d for d in diffs if hard_diff.get(d, {}).get("total", 0) > 0]

    if known_diffs:
        print(f"\n  Per-Difficulty Breakdown (Hard match only)")
        print(f"  {'Difficulty':<15s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
        print(f"  {'-' * 45}")
        for diff in known_diffs:
            data = hard_diff[diff]
            c, t = data["correct"], data["total"]
            print(f"  {diff:<15s} {c:>8d} {t:>8d} {pct(c, t):>10s}")

        print(f"\n  Per-Difficulty Breakdown (All match modes)")
        header = f"  {'Difficulty':<15s}"
        for mode in MATCH_MODES:
            header += f" {mode:>12s}"
        print(header)
        print(f"  {'-' * (15 + 13 * len(MATCH_MODES))}")
        for diff in known_diffs:
            row = f"  {diff:<15s}"
            for mode in MATCH_MODES:
                data = result["diff_accuracy"].get(mode, {}).get(diff, {"correct": 0, "total": 0})
                c, t = data["correct"], data["total"]
                row += f" {pct(c, t):>12s}"
            print(row)
    else:
        print(f"\n  (Per-difficulty breakdown not available for {dataset.upper()})")

    # --- Latency ---
    print(f"\n  Latency (seconds):")
    print(f"  {'Stage':<30s} {'Count':>6s}   Stats")
    print(f"  {'-' * 68}")
    for key in STAGE_KEYS:
        vals = result["stage_latencies"].get(key, [])
        if vals:
            print(f"  {key:<30s} {len(vals):>6d}   {fmt_stat(vals)}")

    # --- Rates ---
    print(f"\n  Pipeline Rates (out of {n_eval} evaluated):")
    print(f"    Revision applied    : {result['revision_count']:>5d}  ({pct(result['revision_count'], n_eval)})")
    print(f"    Refiner applied     : {result['refiner_count']:>5d}  ({pct(result['refiner_count'], n_eval)})")
    print(f"    Verification ran    : {result['verification_count']:>5d}  ({pct(result['verification_count'], n_eval)})")
    print(f"    Verify → regen      : {result['verification_triggered_regen']:>5d}  ({pct(result['verification_triggered_regen'], n_eval)})")


def print_comparison_table(results: List[Dict]):
    if len(results) < 2:
        return

    names = [r["log_name"] for r in results]
    col_w = max(18, max(len(n) for n in names) + 2)
    dataset = results[0]["dataset"]
    diffs = _difficulty_order(dataset)

    print(f"\n{'=' * 80}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 80}")

    header = f"  {'Metric':<28s}" + "".join(f"{n:>{col_w}s}" for n in names)
    print(header)
    print(f"  {'-' * (28 + col_w * len(results))}")

    # Overall accuracy per mode
    for mode in MATCH_MODES:
        row = f"  {f'Overall ({mode})':<28s}"
        for r in results:
            acc = r["accuracy"][mode]
            row += f"{pct(acc['correct'], acc['total']):>{col_w}s}"
        print(row)

    print(f"  {'-' * (28 + col_w * len(results))}")

    # Per-difficulty (hard only, if available)
    known_diffs = [d for d in diffs if any(
        r["diff_accuracy"].get("hard", {}).get(d, {}).get("total", 0) > 0 for r in results
    )]
    for diff in known_diffs:
        row = f"  {diff.capitalize() + ' (hard)':<28s}"
        for r in results:
            data = r["diff_accuracy"].get("hard", {}).get(diff, {"correct": 0, "total": 0})
            row += f"{pct(data['correct'], data['total']):>{col_w}s}"
        print(row)

    print(f"  {'-' * (28 + col_w * len(results))}")

    # Mean total latency
    row = f"  {'Mean total_sec':<28s}"
    for r in results:
        vals = r["stage_latencies"].get("total_sec", [])
        val_str = f"{statistics.mean(vals):.1f}s" if vals else "N/A"
        row += f"{val_str:>{col_w}s}"
    print(row)

    # Mean pipeline total
    row = f"  {'Mean pipeline_total_sec':<28s}"
    for r in results:
        vals = r["stage_latencies"].get("pipeline_total_sec", [])
        val_str = f"{statistics.mean(vals):.1f}s" if vals else "N/A"
        row += f"{val_str:>{col_w}s}"
    print(row)

    # Revision rate
    row = f"  {'Revision rate':<28s}"
    for r in results:
        row += f"{pct(r['revision_count'], r['indices_evaluated']):>{col_w}s}"
    print(row)

    print()


def save_report(results: List[Dict], out_path: Path):
    export = []
    for r in results:
        lat_summary = {}
        for key, vals in r["stage_latencies"].items():
            if vals:
                sorted_vals = sorted(vals)
                lat_summary[key] = {
                    "count": len(vals),
                    "mean": round(statistics.mean(vals), 3),
                    "median": round(statistics.median(vals), 3),
                    "p95": round(sorted_vals[int(len(vals) * 0.95)], 3) if len(vals) >= 2 else round(statistics.mean(vals), 3),
                    "total": round(sum(vals), 3),
                }

        acc_export = {}
        for mode in MATCH_MODES:
            acc = r["accuracy"][mode]
            acc_export[mode] = {
                "correct": acc["correct"],
                "total": acc["total"],
                "accuracy": round(acc["correct"] / acc["total"], 4) if acc["total"] else None,
            }

        diff_acc_export = {}
        for mode in MATCH_MODES:
            diff_acc_export[mode] = {}
            for diff, data in r["diff_accuracy"].get(mode, {}).items():
                c, t = data["correct"], data["total"]
                diff_acc_export[mode][diff] = {
                    "correct": c,
                    "total": t,
                    "accuracy": round(c / t, 4) if t else None,
                }

        export.append({
            "log_name": r["log_name"],
            "log_path": r["log_path"],
            "dataset": r["dataset"],
            "split": r["split"],
            "indices_evaluated": r["indices_evaluated"],
            "subset_missing": r["subset_missing"],
            "exec_errors": r["exec_errors"],
            "gold_errors": r["gold_errors"],
            "accuracy": acc_export,
            "per_difficulty": diff_acc_export,
            "latency": lat_summary,
            "rates": {
                "revision_applied": r["revision_count"],
                "refiner_applied": r["refiner_count"],
                "verification_ran": r["verification_count"],
                "verification_triggered_regen": r["verification_triggered_regen"],
            },
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"\nSaved structured report to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full-dataset execution accuracy + latency evaluation for BIRD/Spider logs"
    )
    parser.add_argument("logs", nargs="+", help="One or more log JSON files to analyse")
    parser.add_argument(
        "--dataset", required=True, choices=["bird", "spider"],
        help="Dataset name: 'bird' or 'spider'"
    )
    parser.add_argument(
        "--split", default="dev",
        help="Dataset split: 'dev', 'train', 'test' (default: dev)"
    )
    parser.add_argument(
        "--index_file", default=None,
        help="Optional JSON file with a list of integer indices to restrict evaluation to"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="SQL execution timeout in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers for SQL execution (default: 8)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save structured JSON report (auto-derived if omitted)"
    )
    args = parser.parse_args()

    # --- Load gold SQL map ---
    print(f"Loading gold SQL for {args.dataset.upper()} / {args.split} ...")
    try:
        gold_map = load_gold_sql_map(args.dataset, args.split)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR loading gold SQL: {e}")
        sys.exit(1)
    print(f"  {len(gold_map)} gold examples loaded")

    # --- Load difficulty map ---
    if args.dataset.lower() == "bird":
        diff_map = load_bird_difficulty_map(args.split)
    else:
        diff_map = load_spider_difficulty_map(args.split)
    if not diff_map:
        print("WARNING: Could not load difficulty map; all examples marked 'unknown'")

    # --- Subset indices ---
    subset_indices: Optional[Set[int]] = None
    if args.index_file:
        idx_path = Path(args.index_file)
        if not idx_path.is_absolute():
            idx_path = BASE_DIR / idx_path
        if not idx_path.exists():
            print(f"ERROR: index file not found: {idx_path}")
            sys.exit(1)
        with open(idx_path, "r") as f:
            subset_indices = set(json.load(f))
        print(f"Restricting to {len(subset_indices)} indices from {idx_path}")
    else:
        print("No index filter — evaluating all examples found in each log")

    # --- Analyse logs ---
    results = []
    for log_arg in args.logs:
        log_path = Path(log_arg)
        if not log_path.is_absolute():
            log_path = BASE_DIR / log_path
        if not log_path.exists():
            print(f"WARNING: log not found: {log_path}, skipping")
            continue
        print(f"\nAnalysing {log_path.name} ...")
        result = analyse_log(
            log_path=log_path,
            dataset=args.dataset,
            split=args.split,
            gold_map=gold_map,
            diff_map=diff_map,
            subset_indices=subset_indices,
            sql_timeout=args.timeout,
            workers=args.workers,
        )
        results.append(result)
        print_report(result)

    if not results:
        print("No logs analysed.")
        sys.exit(1)

    # --- Side-by-side comparison ---
    if len(results) >= 2:
        print_comparison_table(results)

    # --- Save report ---
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = BASE_DIR / out_path
    else:
        tag = f"{args.dataset}_{args.split}_full_eval"
        out_path = (
            BASE_DIR
            / "secondloop"
            / "logs"
            / args.dataset
            / "ablation"
            / f"{tag}_report.json"
        )
    save_report(results, out_path)


if __name__ == "__main__":
    main()
