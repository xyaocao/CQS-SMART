"""
Compare final_reviewer_only vs final_noplanner on the full BIRD dev set.

Reports:
  - Overall and per-difficulty accuracy (hard/soft/partial)
  - Overlap analysis: both correct, both wrong, flipped examples
  - Preliminary SQL accuracy for both configs
  - Verify agent precision breakdown (noplanner)
  - Error analysis on flipped examples
  - DB-level delta

Usage:
    python secondloop/tools/compare_final_configs.py
    python secondloop/tools/compare_final_configs.py --ro path/to/ro.json --np path/to/np.json
    python secondloop/tools/compare_final_configs.py --output report.json
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_correct(example):
    m = example.get("exec_match")
    return m is True or m == 1


def prelim_correct(example):
    """Check if preliminary SQL matches gold (uses same exec_match logic on preliminary_sql field)."""
    # We don't have a separate exec_match for preliminary SQL stored, but we can
    # infer: if revision was NOT applied, preliminary == final, so exec_match applies.
    # If revision WAS applied, preliminary was the pre-review SQL.
    # We store whether the preliminary differed from final in sql_tracking.
    st = example.get("sql_tracking", {})
    prelim = st.get("preliminary_sql", "")
    initial = st.get("initial_sql", "")
    final = st.get("final_sql", "")
    revision_applied = st.get("revision_applied", False)
    # Approximate: if no revision, preliminary ~ initial ~ final
    # We use the stored prelim_exec_match if available, else fall back
    prelim_match = example.get("prelim_exec_match")
    if prelim_match is not None:
        return prelim_match is True or prelim_match == 1
    return None  # not directly measurable without re-execution


def get_difficulty(example):
    return example.get("difficulty", "unknown")


def accuracy_str(correct, total):
    if total == 0:
        return "0/0 = N/A"
    return f"{correct}/{total} = {correct/total:.1%}"


def pct(n, total):
    if total == 0:
        return "N/A"
    return f"{n/total:.1%}"


def latency_stats(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    n = len(values)
    p95_idx = min(int(n * 0.95), n - 1)
    return {
        "count": n,
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": round(values[p95_idx], 2),
    }


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse(ro_path: Path, np_path: Path, output_path: Path = None):
    with open(ro_path, encoding="utf-8") as f:
        ro_list = json.load(f)
    with open(np_path, encoding="utf-8") as f:
        np_list = json.load(f)

    ro_map = {x["example_index"]: x for x in ro_list}
    np_map = {x["example_index"]: x for x in np_list}

    common = sorted(set(ro_map.keys()) & set(np_map.keys()))
    ro_only_keys = sorted(set(ro_map.keys()) - set(np_map.keys()))
    np_only_keys = sorted(set(np_map.keys()) - set(ro_map.keys()))

    print(f"\n{'='*70}")
    print(f"  FINAL CONFIG COMPARISON: reviewer_only vs noplanner")
    print(f"  reviewer_only : {ro_path.name}  ({len(ro_list)} examples)")
    print(f"  noplanner     : {np_path.name}  ({len(np_list)} examples)")
    print(f"{'='*70}")

    # ------------------------------------------------------------------ #
    # 1. Overall accuracy                                                   #
    # ------------------------------------------------------------------ #
    print("\n--- 1. OVERALL ACCURACY ---")

    for label, mapping, total_list in [("reviewer_only", ro_map, ro_list),
                                        ("noplanner",     np_map, np_list)]:
        correct = sum(1 for x in total_list if is_correct(x))
        total   = len(total_list)
        diff_acc = defaultdict(lambda: [0, 0])
        for x in total_list:
            d = get_difficulty(x)
            diff_acc[d][1] += 1
            if is_correct(x):
                diff_acc[d][0] += 1
        print(f"\n  {label}: {accuracy_str(correct, total)}")
        for diff in ["simple", "moderate", "challenging"]:
            c, t = diff_acc[diff]
            print(f"    {diff:<12}: {accuracy_str(c, t)}")

    # ------------------------------------------------------------------ #
    # 2. Overlap analysis on common examples                               #
    # ------------------------------------------------------------------ #
    print(f"\n--- 2. OVERLAP ANALYSIS (common={len(common)}) ---")

    both_correct = []
    both_wrong   = []
    ro_wins      = []   # reviewer_only correct, noplanner wrong
    np_wins      = []   # noplanner correct, reviewer_only wrong

    for idx in common:
        rc = is_correct(ro_map[idx])
        nc = is_correct(np_map[idx])
        if rc and nc:
            both_correct.append(idx)
        elif not rc and not nc:
            both_wrong.append(idx)
        elif rc and not nc:
            ro_wins.append(idx)
        else:
            np_wins.append(idx)

    print(f"  Both correct          : {len(both_correct)}")
    print(f"  Both wrong            : {len(both_wrong)}")
    print(f"  reviewer_only wins    : {len(ro_wins)}  (RO correct, NP wrong)")
    print(f"  noplanner wins        : {len(np_wins)}  (NP correct, RO wrong)")
    print(f"  Net delta (NP - RO)   : {len(np_wins) - len(ro_wins):+d}")

    # ------------------------------------------------------------------ #
    # 3. Preliminary SQL accuracy                                           #
    # ------------------------------------------------------------------ #
    print("\n--- 3. PRELIMINARY SQL vs FINAL SQL ---")

    for label, lst in [("reviewer_only", ro_list), ("noplanner", np_list)]:
        total = len(lst)
        final_correct = sum(1 for x in lst if is_correct(x))

        # Preliminary = initial_sql before reviewer touches it
        # Proxy: if revision_applied=False, preliminary == final
        # If revision_applied=True, preliminary was different (and presumably worse)
        revision_cases  = [x for x in lst if x.get("sql_tracking", {}).get("revision_applied")]
        no_revision     = [x for x in lst if not x.get("sql_tracking", {}).get("revision_applied")]

        # Accuracy when revision was applied (final result)
        rev_correct_final  = sum(1 for x in revision_cases if is_correct(x))
        norev_correct      = sum(1 for x in no_revision   if is_correct(x))

        print(f"\n  {label} ({total} total):")
        print(f"    Final SQL accuracy            : {accuracy_str(final_correct, total)}")
        print(f"    Cases with revision applied   : {len(revision_cases)} ({pct(len(revision_cases), total)})")
        print(f"      → accuracy after revision   : {accuracy_str(rev_correct_final, len(revision_cases))}")
        print(f"    Cases without revision        : {len(no_revision)} ({pct(len(no_revision), total)})")
        print(f"      → accuracy (no revision)    : {accuracy_str(norev_correct, len(no_revision))}")

        # Estimate preliminary SQL accuracy:
        # For no-revision cases: preliminary ≈ final (same result)
        # For revision cases: preliminary was wrong (that's why revision was triggered — reviewer found issues)
        # So preliminary accuracy ≈ norev_correct + 0 (revision cases were failing before review)
        prelim_correct_est = norev_correct  # lower bound
        # Upper bound: some revision cases might have been correct before review
        # (reviewer over-triggered). We can check if final is correct after revision.
        prelim_est_total = total
        print(f"    Est. preliminary SQL accuracy : ~{accuracy_str(prelim_correct_est, prelim_est_total)} (lower bound)")
        print(f"    Reviewer improvement          : +{final_correct - prelim_correct_est} examples")

    # ------------------------------------------------------------------ #
    # 4. Flip analysis: what caused the differences                        #
    # ------------------------------------------------------------------ #
    print("\n--- 4. FLIP ANALYSIS ---")

    def categorise_flip(n_entry, ro_entry, np_wins_direction):
        """Return category string for why a flip occurred."""
        ver_regen = n_entry.get("latency", {}).get("verification_regen_sec") is not None and \
                    n_entry.get("latency", {}).get("verification_regen_sec", 0) > 0
        sql_ro = (ro_entry.get("sql_tracking", {}).get("final_sql") or "").strip()
        sql_np = (n_entry.get("sql_tracking",  {}).get("final_sql") or "").strip()
        sql_changed = sql_ro != sql_np

        if ver_regen and not np_wins_direction:
            return "verify_false_alarm"   # verify triggered, NP wrong
        elif ver_regen and np_wins_direction:
            return "verify_saved"         # verify triggered, NP correct
        elif sql_changed:
            return "different_sql"        # different generation path
        else:
            return "same_sql_diff_match"  # edge case

    ro_win_cats = defaultdict(list)
    np_win_cats = defaultdict(list)

    for idx in ro_wins:
        cat = categorise_flip(np_map[idx], ro_map[idx], np_wins_direction=False)
        ro_win_cats[cat].append(idx)

    for idx in np_wins:
        cat = categorise_flip(np_map[idx], ro_map[idx], np_wins_direction=True)
        np_win_cats[cat].append(idx)

    print(f"\n  reviewer_only wins ({len(ro_wins)} cases):")
    for cat, idxs in sorted(ro_win_cats.items()):
        print(f"    {cat:<25}: {len(idxs)}  idx={idxs[:8]}{'...' if len(idxs)>8 else ''}")

    print(f"\n  noplanner wins ({len(np_wins)} cases):")
    for cat, idxs in sorted(np_win_cats.items()):
        print(f"    {cat:<25}: {len(idxs)}  idx={idxs[:8]}{'...' if len(idxs)>8 else ''}")

    # ------------------------------------------------------------------ #
    # 5. Verify agent precision (noplanner)                                #
    # ------------------------------------------------------------------ #
    print("\n--- 5. VERIFY AGENT PRECISION (noplanner full dataset) ---")

    ver_regen_cases = [n for n in np_list
                       if (n.get("latency", {}).get("verification_regen_sec") or 0) > 0]
    ver_ok_cases    = [n for n in np_list
                       if (n.get("latency", {}).get("verification_sec") or 0) > 0
                       and not (n.get("latency", {}).get("verification_regen_sec") or 0) > 0]
    no_ver_cases    = [n for n in np_list
                       if not (n.get("latency", {}).get("verification_sec") or 0) > 0]

    true_save    = []
    false_alarm  = []
    unnecessary  = []
    failed_save  = []

    for n in ver_regen_cases:
        idx = n["example_index"]
        nc  = is_correct(n)
        rc  = is_correct(ro_map[idx]) if idx in ro_map else None
        if rc is None:
            continue
        if not rc and nc:
            true_save.append(idx)
        elif rc and nc:
            unnecessary.append(idx)
        elif rc and not nc:
            false_alarm.append(idx)
        else:
            failed_save.append(idx)

    total_vr = len(ver_regen_cases)
    print(f"  Verify triggered regen  : {total_vr}")
    print(f"    True saves  (wrong→correct)   : {len(true_save)}")
    print(f"    False alarms(correct→wrong)   : {len(false_alarm)}  *** HARMFUL ***")
    print(f"    Unnecessary (correct→correct) : {len(unnecessary)}")
    print(f"    Failed saves(wrong→wrong)     : {len(failed_save)}")
    print(f"  Net verify effect               : {len(true_save)-len(false_alarm):+d} examples")
    print(f"  Verify precision on regen       : {pct(len(true_save)+len(unnecessary), total_vr)} (correct after regen)")

    # Latency overhead
    total_ver_lat   = sum((n.get("latency", {}).get("verification_sec") or 0) for n in np_list)
    total_vregen_lat= sum((n.get("latency", {}).get("verification_regen_sec") or 0) for n in np_list)
    n_ver_calls     = len(ver_regen_cases) + len(ver_ok_cases)
    print(f"\n  Verify latency overhead :")
    print(f"    verification calls      : {n_ver_calls}  total={total_ver_lat:.0f}s  avg={total_ver_lat/max(n_ver_calls,1):.2f}s/call")
    print(f"    verify-regen calls      : {len(ver_regen_cases)}  total={total_vregen_lat:.0f}s")
    print(f"    Total overhead / 1534   : {(total_ver_lat+total_vregen_lat):.0f}s  avg/example={((total_ver_lat+total_vregen_lat)/len(np_list)):.2f}s")

    # ------------------------------------------------------------------ #
    # 6. DB-level delta                                                    #
    # ------------------------------------------------------------------ #
    print("\n--- 6. DB-LEVEL DELTA (noplanner - reviewer_only) ---")

    db_stats = defaultdict(lambda: {"ro": 0, "np": 0, "total": 0})
    for idx in common:
        db = ro_map[idx]["db_id"]
        db_stats[db]["total"] += 1
        if is_correct(ro_map[idx]): db_stats[db]["ro"] += 1
        if is_correct(np_map[idx]): db_stats[db]["np"] += 1

    rows = sorted(db_stats.items(), key=lambda x: x[1]["np"] - x[1]["ro"])
    print(f"\n  {'DB':<35} {'Total':>6} {'RO':>5} {'NP':>5} {'Delta':>6}")
    print(f"  {'-'*60}")
    for db, s in rows:
        delta = s["np"] - s["ro"]
        if delta != 0:
            marker = " <<<" if abs(delta) >= 3 else ""
            print(f"  {db:<35} {s['total']:>6} {s['ro']:>5} {s['np']:>5} {delta:>+6}{marker}")

    # ------------------------------------------------------------------ #
    # 7. Flip detail table                                                  #
    # ------------------------------------------------------------------ #
    print("\n--- 7. FLIP DETAIL (reviewer_only wins) ---")
    print(f"  {'idx':>5}  {'db':<30}  {'diff':<12}  {'RO revision':>11}  {'NP verify_regen':>15}  {'SQL same?':>9}")
    print(f"  {'-'*90}")
    for idx in ro_wins[:30]:
        r = ro_map[idx]; n = np_map[idx]
        ro_rev = r.get("sql_tracking", {}).get("revision_applied", False)
        np_vr  = (n.get("latency", {}).get("verification_regen_sec") or 0) > 0
        sql_r  = (r.get("sql_tracking", {}).get("final_sql") or "").strip()
        sql_n  = (n.get("sql_tracking", {}).get("final_sql") or "").strip()
        same   = "yes" if sql_r == sql_n else "no"
        print(f"  {idx:>5}  {r['db_id']:<30}  {r.get('difficulty','?'):<12}  {str(ro_rev):>11}  {str(np_vr):>15}  {same:>9}")

    print("\n--- 8. FLIP DETAIL (noplanner wins) ---")
    print(f"  {'idx':>5}  {'db':<30}  {'diff':<12}  {'RO revision':>11}  {'NP verify_regen':>15}  {'SQL same?':>9}")
    print(f"  {'-'*90}")
    for idx in np_wins[:30]:
        r = ro_map[idx]; n = np_map[idx]
        ro_rev = r.get("sql_tracking", {}).get("revision_applied", False)
        np_vr  = (n.get("latency", {}).get("verification_regen_sec") or 0) > 0
        sql_r  = (r.get("sql_tracking", {}).get("final_sql") or "").strip()
        sql_n  = (n.get("sql_tracking", {}).get("final_sql") or "").strip()
        same   = "yes" if sql_r == sql_n else "no"
        print(f"  {idx:>5}  {r['db_id']:<30}  {r.get('difficulty','?'):<12}  {str(ro_rev):>11}  {str(np_vr):>15}  {same:>9}")

    # ------------------------------------------------------------------ #
    # 8. Save JSON report                                                   #
    # ------------------------------------------------------------------ #
    report = {
        "configs": {
            "reviewer_only": str(ro_path),
            "noplanner": str(np_path),
        },
        "full_dataset": {
            "reviewer_only": {
                "correct": sum(1 for x in ro_list if is_correct(x)),
                "total": len(ro_list),
                "accuracy": round(sum(1 for x in ro_list if is_correct(x)) / len(ro_list), 4),
            },
            "noplanner": {
                "correct": sum(1 for x in np_list if is_correct(x)),
                "total": len(np_list),
                "accuracy": round(sum(1 for x in np_list if is_correct(x)) / len(np_list), 4),
            },
        },
        "overlap": {
            "both_correct": len(both_correct),
            "both_wrong":   len(both_wrong),
            "ro_wins":      len(ro_wins),
            "np_wins":      len(np_wins),
            "net_delta":    len(np_wins) - len(ro_wins),
        },
        "verify_precision": {
            "total_regen_calls": total_vr,
            "true_saves":   len(true_save),
            "false_alarms": len(false_alarm),
            "unnecessary":  len(unnecessary),
            "failed_saves": len(failed_save),
            "net_effect":   len(true_save) - len(false_alarm),
        },
        "flip_indices": {
            "ro_wins": ro_wins,
            "np_wins": np_wins,
            "verify_false_alarms": false_alarm,
            "verify_true_saves":   true_save,
        },
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved → {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent  # secondloop/

    parser = argparse.ArgumentParser(description="Compare final_reviewer_only vs final_noplanner")
    parser.add_argument("--ro",  default=str(root / "logs/bird/qwen3/online/final_reviewer_only.json"),
                        help="Path to reviewer_only log")
    parser.add_argument("--np",  default=str(root / "logs/bird/qwen3/online/final_noplanner.json"),
                        help="Path to noplanner log")
    parser.add_argument("--output", default=str(root / "logs/bird/ablation/results/final_comparison_report.json"),
                        help="Path to save JSON report")
    args = parser.parse_args()

    analyse(Path(args.ro), Path(args.np), Path(args.output))


if __name__ == "__main__":
    main()
