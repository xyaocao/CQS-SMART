"""
Compare ablation study results on the 500-example BIRD subset.

Reads one or more log files, filters to the 500 subset indices, and reports:
  - Overall accuracy
  - Per-difficulty accuracy breakdown
  - Latency statistics (mean, median, p95)
  - Stage-level latency breakdown
  - Revision / refiner / verification rates

Usage:
    # Single log
    python secondloop/tools/compare_ablation.py secondloop/logs/bird/qwen3/online/reviewer_only.json

    # Compare multiple logs side by side
    python secondloop/tools/compare_ablation.py \
        secondloop/logs/bird/qwen3/online/run1.json \
        secondloop/logs/bird/qwen3/online/SQLGenOnly.json

    # Use custom index file
    python secondloop/tools/compare_ablation.py --index_file my_indices.json log1.json log2.json

    # Skip index filtering (use all examples in log)
    python secondloop/tools/compare_ablation.py --no_filter log1.json
"""
import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_FILE = Path(__file__).resolve().parent / "bird_dev_500.json"
BIRD_DEV_PATH = BASE_DIR / "Data" / "BIRD" / "dev" / "dev.json"

# Latency keys that represent actual LLM / pipeline durations (seconds)
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


def load_difficulty_map() -> dict[int, str]:
    """Map example index -> difficulty from BIRD dev.json."""
    if not BIRD_DEV_PATH.exists():
        return {}
    with open(BIRD_DEV_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {i: item.get("difficulty", "unknown").lower() for i, item in enumerate(data)}


def pct(n: int, d: int) -> str:
    return f"{n/d*100:.1f}%" if d > 0 else "N/A"


def fmt_stat(values: list[float]) -> str:
    """Format mean / median / p95 for a list of float values."""
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    med = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 2 else mean
    return f"mean={mean:.1f}s  med={med:.1f}s  p95={p95:.1f}s"


def analyse_log(log_path: Path, subset_indices: set[int] | None, diff_map: dict[int, str]) -> dict:
    """Analyse a single log file filtered to subset_indices."""
    with open(log_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = [records]

    # Index records by example_index
    by_idx: dict[int, dict] = {}
    for r in records:
        idx = r.get("example_index")
        if idx is not None:
            by_idx[idx] = r  # last entry wins (in case of duplicates)

    # Filter to subset
    if subset_indices is not None:
        indices = sorted(i for i in subset_indices if i in by_idx)
        missing = sorted(i for i in subset_indices if i not in by_idx)
    else:
        indices = sorted(by_idx.keys())
        missing = []

    # --- Accuracy ---
    overall_correct = 0
    overall_total = 0
    diff_correct: dict[str, int] = defaultdict(int)
    diff_total: dict[str, int] = defaultdict(int)

    # --- Latency ---
    stage_latencies: dict[str, list[float]] = defaultdict(list)

    # --- Rates ---
    revision_count = 0
    refiner_count = 0
    verification_count = 0
    verification_triggered_regen = 0

    for idx in indices:
        r = by_idx[idx]
        match = r.get("exec_match", False)
        diff = diff_map.get(idx, "unknown")

        overall_total += 1
        if match:
            overall_correct += 1
        diff_total[diff] += 1
        if match:
            diff_correct[diff] += 1

        # Latency
        lat = r.get("latency", {})
        for key in STAGE_KEYS:
            val = lat.get(key)
            if val is not None and isinstance(val, (int, float)) and val > 0:
                stage_latencies[key].append(float(val))

        # Rates
        st = r.get("sql_tracking", {})
        if st.get("revision_applied"):
            revision_count += 1
        if st.get("refiner_applied"):
            refiner_count += 1
        if st.get("verification_applied"):
            verification_count += 1
        dbg = r.get("debug", {})
        if dbg and dbg.get("verification_triggered_regen"):
            verification_triggered_regen += 1

    return {
        "log_path": str(log_path),
        "log_name": log_path.stem,
        "total_in_log": len(records),
        "subset_matched": len(indices),
        "subset_missing": len(missing),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy": overall_correct / overall_total if overall_total else 0,
        "diff_correct": dict(diff_correct),
        "diff_total": dict(diff_total),
        "stage_latencies": {k: v for k, v in stage_latencies.items()},
        "revision_count": revision_count,
        "refiner_count": refiner_count,
        "verification_count": verification_count,
        "verification_triggered_regen": verification_triggered_regen,
    }


def print_report(result: dict):
    """Pretty-print a single log analysis."""
    print(f"\n{'=' * 65}")
    print(f"  Log: {result['log_name']}")
    print(f"  ({result['log_path']})")
    print(f"{'=' * 65}")

    n = result["overall_total"]
    c = result["overall_correct"]
    print(f"\n  Overall Accuracy: {c}/{n} = {pct(c, n)}")
    if result["subset_missing"] > 0:
        print(f"  WARNING: {result['subset_missing']} subset indices not found in log")

    # Per-difficulty
    print(f"\n  Per-Difficulty Breakdown:")
    print(f"  {'Difficulty':<15s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print(f"  {'-'*43}")
    for diff in ["simple", "moderate", "challenging"]:
        dt = result["diff_total"].get(diff, 0)
        dc = result["diff_correct"].get(diff, 0)
        print(f"  {diff:<15s} {dc:>8d} {dt:>8d} {pct(dc, dt):>10s}")

    # Latency
    print(f"\n  Latency (seconds):")
    print(f"  {'Stage':<28s} {'Count':>6s}   {'Stats'}")
    print(f"  {'-'*65}")
    for key in STAGE_KEYS:
        vals = result["stage_latencies"].get(key, [])
        if vals:
            print(f"  {key:<28s} {len(vals):>6d}   {fmt_stat(vals)}")

    # Rates
    print(f"\n  Pipeline Rates (out of {n}):")
    print(f"    Revision applied:     {result['revision_count']:>5d}  ({pct(result['revision_count'], n)})")
    print(f"    Refiner applied:      {result['refiner_count']:>5d}  ({pct(result['refiner_count'], n)})")
    print(f"    Verification ran:     {result['verification_count']:>5d}  ({pct(result['verification_count'], n)})")
    print(f"    Verify→regen:         {result['verification_triggered_regen']:>5d}  ({pct(result['verification_triggered_regen'], n)})")


def print_comparison_table(results: list[dict]):
    """Print a side-by-side comparison table for multiple logs."""
    if len(results) < 2:
        return

    print(f"\n{'=' * 75}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 75}")

    # Header
    names = [r["log_name"] for r in results]
    col_w = max(18, max(len(n) for n in names) + 2)
    header = f"  {'Metric':<22s}" + "".join(f"{n:>{col_w}s}" for n in names)
    print(header)
    print(f"  {'-' * (22 + col_w * len(names))}")

    # Overall
    row = f"  {'Overall Accuracy':<22s}"
    for r in results:
        row += f"{pct(r['overall_correct'], r['overall_total']):>{col_w}s}"
    print(row)

    # Per-difficulty
    for diff in ["simple", "moderate", "challenging"]:
        row = f"  {diff.capitalize():<22s}"
        for r in results:
            dt = r["diff_total"].get(diff, 0)
            dc = r["diff_correct"].get(diff, 0)
            row += f"{pct(dc, dt):>{col_w}s}"
        print(row)

    # Mean total latency
    row = f"  {'Mean total_sec':<22s}"
    for r in results:
        vals = r["stage_latencies"].get("total_sec", [])
        val_str = f"{statistics.mean(vals):.1f}s" if vals else "N/A"
        row += f"{val_str:>{col_w}s}"
    print(row)

    # Revision rate
    row = f"  {'Revision rate':<22s}"
    for r in results:
        row += f"{pct(r['revision_count'], r['overall_total']):>{col_w}s}"
    print(row)

    print()


def save_report(results: list[dict], out_path: Path):
    """Save structured results to JSON."""
    export = []
    for r in results:
        # Convert latency lists to summary stats
        lat_summary = {}
        for key, vals in r["stage_latencies"].items():
            if vals:
                lat_summary[key] = {
                    "count": len(vals),
                    "mean": round(statistics.mean(vals), 2),
                    "median": round(statistics.median(vals), 2),
                    "p95": round(sorted(vals)[int(len(vals) * 0.95)], 2) if len(vals) >= 2 else round(statistics.mean(vals), 2),
                    "total": round(sum(vals), 2),
                }

        diff_accuracy = {}
        for diff in ["simple", "moderate", "challenging"]:
            dt = r["diff_total"].get(diff, 0)
            dc = r["diff_correct"].get(diff, 0)
            diff_accuracy[diff] = {
                "correct": dc,
                "total": dt,
                "accuracy": round(dc / dt, 4) if dt > 0 else None,
            }

        export.append({
            "log_name": r["log_name"],
            "log_path": r["log_path"],
            "subset_matched": r["subset_matched"],
            "subset_missing": r["subset_missing"],
            "overall": {
                "correct": r["overall_correct"],
                "total": r["overall_total"],
                "accuracy": round(r["overall_accuracy"], 4),
            },
            "per_difficulty": diff_accuracy,
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


def main():
    parser = argparse.ArgumentParser(description="Compare ablation results on BIRD 500 subset")
    parser.add_argument("logs", nargs="+", help="One or more log JSON files to analyse")
    parser.add_argument("--index_file", type=str, default=None,
                        help="JSON file with subset indices (default: bird_dev_500.json)")
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip index filtering, use all examples in log")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save structured JSON report")
    args = parser.parse_args()

    # Load subset indices
    if args.no_filter:
        subset_indices = None
        print("No index filtering — using all examples in each log")
    else:
        idx_path = Path(args.index_file) if args.index_file else DEFAULT_INDEX_FILE
        if not idx_path.is_absolute():
            idx_path = BASE_DIR / idx_path
        if not idx_path.exists():
            print(f"ERROR: Index file not found: {idx_path}")
            return
        with open(idx_path, "r") as f:
            subset_indices = set(json.load(f))
        print(f"Loaded {len(subset_indices)} subset indices from {idx_path}")

    # Load difficulty map
    diff_map = load_difficulty_map()
    if not diff_map:
        print("WARNING: Could not load BIRD dev.json for difficulty mapping")

    # Analyse each log
    results = []
    for log_arg in args.logs:
        log_path = Path(log_arg)
        if not log_path.is_absolute():
            log_path = BASE_DIR / log_path
        if not log_path.exists():
            print(f"WARNING: Log not found: {log_path}, skipping")
            continue
        result = analyse_log(log_path, subset_indices, diff_map)
        results.append(result)
        print_report(result)

    # Side-by-side comparison
    if len(results) >= 2:
        print_comparison_table(results)

    # Save report
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = BASE_DIR / out_path
        save_report(results, out_path)
    else:
        # Auto-save next to the first log
        out_path = BASE_DIR / "secondloop" / "logs" / "bird" / "ablation" / "comparison_report.json"
        save_report(results, out_path)


if __name__ == "__main__":
    main()
