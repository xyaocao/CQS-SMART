"""
Reviewer Impact Analysis: within a single log, re-execute initial_sql against
the gold standard and compare with final_sql to measure how much the reviewer
loop helps vs hurts.

Produces:
  - wrong→correct  (reviewer saved it)
  - correct→wrong  (reviewer hurt it)
  - both correct / both wrong
  - per-difficulty breakdown
  - revision_applied sub-breakdown (when revision happened vs when it didn't)

Usage:
    python secondloop/tools/reviewer_impact.py \
        --log secondloop/logs/bird/qwen3/online/final_reviewer_only.json \
        --output secondloop/logs/bird/ablation/results/reviewer_impact_final.txt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BIRD_DEV_PATH = BASE_DIR / "Data" / "BIRD" / "dev" / "dev.json"
DB_ROOT = BASE_DIR / "Data" / "BIRD" / "dev" / "dev_databases"

sys.path.insert(0, str(BASE_DIR))
from baseline.exec_match import exec_sql, exec_match, resolve_db_path


# ── helpers ──────────────────────────────────────────────────────────

def load_difficulty_map() -> dict:
    if not BIRD_DEV_PATH.exists():
        return {}
    with open(BIRD_DEV_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {i: item.get("difficulty", "unknown").lower() for i, item in enumerate(data)}


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n/d*100:.1f}%)" if d > 0 else "0/0 (N/A)"


def exec_initial(db_path: str, initial_sql: str, gold_sql: str):
    """Execute initial_sql and gold_sql, return (is_match, error)."""
    try:
        init_rows = exec_sql(db_path, initial_sql)
        gold_rows = exec_sql(db_path, gold_sql)
        is_match, _ = exec_match(init_rows, gold_rows, order_matters=False, match_mode="hard")
        return is_match, None
    except Exception as e:
        return False, str(e)


# ── core ─────────────────────────────────────────────────────────────

def analyse(log_path: Path, diff_map: dict) -> dict:
    with open(log_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = [records]

    # counters
    both_correct = 0
    both_wrong = 0
    wrong_to_right = []   # reviewer saved
    right_to_wrong = []   # reviewer hurt
    exec_errors = 0

    # sub-breakdown by revision_applied
    no_rev_both_correct = 0
    no_rev_both_wrong = 0
    rev_wrong_to_right = 0
    rev_right_to_wrong = 0
    rev_both_correct = 0
    rev_both_wrong = 0

    # per-difficulty
    diff_w2r = defaultdict(int)
    diff_r2w = defaultdict(int)

    total = len(records)
    print(f"Evaluating {total} records...")

    for i, r in enumerate(records):
        if i % 100 == 0:
            print(f"  {i}/{total}...", end="\r")

        idx = r.get("example_index", i)
        db_id = r.get("db_id", "")
        st = r.get("sql_tracking", {}) or {}
        lat = r.get("latency", {}) or {}

        initial_sql = st.get("initial_sql", "")
        final_sql = st.get("final_sql", "")
        revision_applied = bool(st.get("revision_applied", False))
        final_match = bool(r.get("exec_match", False))
        gold_sql = lat.get("gold_sql_executed", "")
        difficulty = diff_map.get(idx, "unknown")

        if not initial_sql or not gold_sql:
            exec_errors += 1
            continue

        db_path = resolve_db_path(str(DB_ROOT), db_id, split="dev", dataset="bird")
        if not db_path:
            exec_errors += 1
            continue

        init_match, err = exec_initial(db_path, initial_sql, gold_sql)
        if err and not init_match:
            # execution error on initial SQL — count as wrong
            pass

        entry = {
            "example_index": idx,
            "question": r.get("question", ""),
            "db_id": db_id,
            "difficulty": difficulty,
            "revision_applied": revision_applied,
            "initial_sql": initial_sql,
            "final_sql": final_sql,
            "review_verdict": (r.get("review") or {}).get("verdict", ""),
            "review_issues": (r.get("review") or {}).get("issues", []),
        }

        if init_match and final_match:
            both_correct += 1
            if revision_applied:
                rev_both_correct += 1
            else:
                no_rev_both_correct += 1
        elif not init_match and not final_match:
            both_wrong += 1
            if revision_applied:
                rev_both_wrong += 1
            else:
                no_rev_both_wrong += 1
        elif not init_match and final_match:
            wrong_to_right.append(entry)
            diff_w2r[difficulty] += 1
            if revision_applied:
                rev_wrong_to_right += 1
        else:  # init_match and not final_match
            right_to_wrong.append(entry)
            diff_r2w[difficulty] += 1
            if revision_applied:
                rev_right_to_wrong += 1

    print(f"\nDone. Errors/skipped: {exec_errors}")

    evaluated = total - exec_errors
    initial_correct = both_correct + len(right_to_wrong)
    final_correct = both_correct + len(wrong_to_right)

    return {
        "log": str(log_path),
        "total_records": total,
        "evaluated": evaluated,
        "exec_errors": exec_errors,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "revision_breakdown": {
            "no_revision_both_correct": no_rev_both_correct,
            "no_revision_both_wrong": no_rev_both_wrong,
            "revision_wrong_to_right": rev_wrong_to_right,
            "revision_right_to_wrong": rev_right_to_wrong,
            "revision_both_correct": rev_both_correct,
            "revision_both_wrong": rev_both_wrong,
        },
        "per_difficulty": {
            diff: {
                "wrong_to_right": diff_w2r.get(diff, 0),
                "right_to_wrong": diff_r2w.get(diff, 0),
                "net": diff_w2r.get(diff, 0) - diff_r2w.get(diff, 0),
            }
            for diff in ["simple", "moderate", "challenging"]
        },
    }


# ── report ───────────────────────────────────────────────────────────

def format_report(result: dict) -> str:
    lines = []
    ev = result["evaluated"]
    ic = result["initial_correct"]
    fc = result["final_correct"]
    w2r = len(result["wrong_to_right"])
    r2w = len(result["right_to_wrong"])
    bc = result["both_correct"]
    bw = result["both_wrong"]
    rb = result["revision_breakdown"]

    lines.append("=" * 65)
    lines.append("  REVIEWER IMPACT ANALYSIS")
    lines.append(f"  Log: {Path(result['log']).name}")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"  Total records   : {result['total_records']}")
    lines.append(f"  Evaluated       : {ev}")
    lines.append(f"  Exec errors     : {result['exec_errors']}")
    lines.append("")
    lines.append("  ACCURACY")
    lines.append(f"  {'Initial SQL':30s}: {pct(ic, ev)}")
    lines.append(f"  {'Final SQL':30s}: {pct(fc, ev)}")
    lines.append(f"  {'Net gain':30s}: {fc - ic:+d} examples")
    lines.append("")
    lines.append("  TRANSITION COUNTS")
    lines.append(f"  {'Both correct (unchanged)':30s}: {bc}")
    lines.append(f"  {'Both wrong (unchanged)':30s}: {bw}")
    lines.append(f"  {'Wrong → Correct (helped)':30s}: {w2r}")
    lines.append(f"  {'Correct → Wrong (hurt)':30s}: {r2w}")
    lines.append(f"  {'Net':30s}: {w2r - r2w:+d}")
    lines.append("")
    lines.append("  REVISION SUB-BREAKDOWN")
    lines.append(f"  {'':5s}{'No revision applied':30s}")
    lines.append(f"  {'':5s}  Both correct      : {rb['no_revision_both_correct']}")
    lines.append(f"  {'':5s}  Both wrong        : {rb['no_revision_both_wrong']}")
    lines.append(f"  {'':5s}{'Revision applied':30s}")
    lines.append(f"  {'':5s}  Wrong → Correct   : {rb['revision_wrong_to_right']}")
    lines.append(f"  {'':5s}  Correct → Wrong   : {rb['revision_right_to_wrong']}")
    lines.append(f"  {'':5s}  Both correct      : {rb['revision_both_correct']}")
    lines.append(f"  {'':5s}  Both wrong        : {rb['revision_both_wrong']}")
    lines.append("")
    lines.append("  PER-DIFFICULTY BREAKDOWN")
    lines.append(f"  {'Difficulty':<15s} {'Helped':>8s} {'Hurt':>8s} {'Net':>8s}")
    lines.append("  " + "-" * 40)
    for diff in ["simple", "moderate", "challenging"]:
        d = result["per_difficulty"][diff]
        lines.append(
            f"  {diff:<15s} {d['wrong_to_right']:>8d} {d['right_to_wrong']:>8d} "
            f"{d['net']:>+8d}"
        )
    lines.append("")
    lines.append("  WRONG → CORRECT EXAMPLES (reviewer helped)")
    lines.append("  " + "-" * 65)
    for e in result["wrong_to_right"]:
        rev = "R" if e["revision_applied"] else "-"
        lines.append(
            f"  [{e['example_index']:>4d}] {e['difficulty']:<12s} [{rev}] "
            f"{e['db_id']:<30s} {e['question'][:40]}"
        )
    lines.append("")
    lines.append("  CORRECT → WRONG EXAMPLES (reviewer hurt)")
    lines.append("  " + "-" * 65)
    for e in result["right_to_wrong"]:
        rev = "R" if e["revision_applied"] else "-"
        verdict = e.get("review_verdict", "")
        lines.append(
            f"  [{e['example_index']:>4d}] {e['difficulty']:<12s} [{rev}] "
            f"{e['db_id']:<30s} verdict={verdict}"
        )
    lines.append("")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reviewer impact: initial vs final SQL accuracy")
    parser.add_argument("--log", required=True, help="Path to the log JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write the text report (default: ablation/results/)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = BASE_DIR / log_path
    if not log_path.exists():
        print(f"ERROR: Log not found: {log_path}")
        sys.exit(1)

    diff_map = load_difficulty_map()
    result = analyse(log_path, diff_map)

    report = format_report(result)
    print(report)

    # save text report
    if args.output:
        out_txt = Path(args.output)
    else:
        stem = log_path.stem
        out_txt = (
            BASE_DIR / "secondloop" / "logs" / "bird" / "ablation" / "results"
            / f"reviewer_impact_{stem}.txt"
        )
    if not out_txt.is_absolute():
        out_txt = BASE_DIR / out_txt
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {out_txt}")

    # save JSON with full detail
    out_json = out_txt.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Detail JSON saved to {out_json}")


if __name__ == "__main__":
    main()
