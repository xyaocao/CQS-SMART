"""
Generate a consolidated ablation summary table for the thesis.

Reads all available ablation log files, computes accuracy + McNemar's test
for each configuration pair, and outputs a JSON summary suitable for LaTeX
table generation.

Usage:
    # Default: 500-subset results
    python secondloop/tools/ablation_summary.py

    # Include full-dataset results (if available)
    python secondloop/tools/ablation_summary.py --include_full

    # Custom output path
    python secondloop/tools/ablation_summary.py --output path/to/summary.json
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_FILE = Path(__file__).resolve().parent / "bird_dev_500.json"
BIRD_DEV_PATH = BASE_DIR / "Data" / "BIRD" / "dev" / "dev.json"

# ── Configuration registry ──────────────────────────────────────────
# Maps config name -> (log path relative to BASE_DIR, description, flags)
SUBSET_CONFIGS = {
    "C7_SQLGenOnly": {
        "path": "secondloop/logs/bird/qwen3/online/SQLGenOnly.json",
        "description": "SQLGen only (no planner, reviewer, or verifier)",
        "flags": "online_schema",
    },
    "C1_full_pipeline": {
        "path": "secondloop/logs/bird/qwen3/online/run1.json",
        "description": "Full pipeline (planner + reviewer + verifier)",
        "flags": "online_schema, refine, verify",
    },
    "noplanner": {
        "path": "secondloop/logs/bird/qwen3/online/noplanner.json",
        "description": "Reviewer + verifier (no planner)",
        "flags": "online_schema, skip_planner, refine, verify",
    },
    "offline_original": {
        "path": "secondloop/logs/bird/qwen3/offline/run_log.json",
        "description": "Offline schema (PPL + step2), original engine",
        "flags": "use_ppl_schema, refine, verify (original engine)",
    },
    "offline_no_step2": {
        "path": "secondloop/logs/bird/qwen3/offline/run_no_step2_500.json",
        "description": "Offline schema (PPL, no step2 keywords/conditions), current engine",
        "flags": "use_ppl_schema, refine, verify",
    },
}

FULL_CONFIGS = {
    "C7_SQLGenOnly_full": {
        "path": "secondloop/logs/bird/qwen3/online/full_SQLGenOnly.json",
        "description": "SQLGen only — full 1534 dev set",
        "flags": "online_schema",
    },
    "C1_full_pipeline_full": {
        "path": "secondloop/logs/bird/qwen3/online/full_run1.json",
        "description": "Full pipeline — full 1534 dev set",
        "flags": "online_schema, refine, verify",
    },
}


def load_difficulty_map() -> dict[int, str]:
    if not BIRD_DEV_PATH.exists():
        return {}
    with open(BIRD_DEV_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {i: item.get("difficulty", "unknown").lower() for i, item in enumerate(data)}


def load_log(path: Path) -> dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = [records]
    return {r["example_index"]: r for r in records if "example_index" in r}


def compute_accuracy(by_idx: dict[int, dict], indices: set[int], diff_map: dict[int, str]) -> dict:
    """Compute overall and per-difficulty accuracy."""
    correct = 0
    total = 0
    diff_correct = defaultdict(int)
    diff_total = defaultdict(int)

    for idx in sorted(indices):
        rec = by_idx.get(idx)
        if rec is None:
            continue
        total += 1
        match = rec.get("exec_match", False)
        diff = diff_map.get(idx, "unknown")
        diff_total[diff] += 1
        if match:
            correct += 1
            diff_correct[diff] += 1

    per_diff = {}
    for d in ["simple", "moderate", "challenging"]:
        dt = diff_total.get(d, 0)
        dc = diff_correct.get(d, 0)
        per_diff[d] = {
            "correct": dc,
            "total": dt,
            "accuracy": round(dc / dt, 4) if dt > 0 else None,
        }

    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total > 0 else None,
        "per_difficulty": per_diff,
    }


def mcnemar_test(by_idx_a: dict, by_idx_b: dict, indices: set[int]) -> dict:
    """Compute McNemar's test between two configs on shared indices."""
    broken = 0  # A correct, B wrong
    fixed = 0   # A wrong, B correct

    for idx in indices:
        rec_a = by_idx_a.get(idx)
        rec_b = by_idx_b.get(idx)
        if rec_a is None or rec_b is None:
            continue
        a_match = rec_a.get("exec_match", False)
        b_match = rec_b.get("exec_match", False)
        if a_match and not b_match:
            broken += 1
        elif not a_match and b_match:
            fixed += 1

    discordant = broken + fixed
    if discordant > 0:
        chi2 = (broken - fixed) ** 2 / discordant
        significant = chi2 > 3.841
    else:
        chi2 = 0.0
        significant = False

    return {
        "broken": broken,
        "fixed": fixed,
        "discordant": discordant,
        "chi2": round(chi2, 4),
        "significant": significant,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ablation summary for thesis")
    parser.add_argument("--include_full", action="store_true",
                        help="Include full-dataset results if available")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip index filtering for subset configs")
    args = parser.parse_args()

    diff_map = load_difficulty_map()

    # Load subset indices
    if args.no_filter:
        subset_indices = None
    else:
        if DEFAULT_INDEX_FILE.exists():
            with open(DEFAULT_INDEX_FILE) as f:
                subset_indices = set(json.load(f))
            print(f"Loaded {len(subset_indices)} subset indices")
        else:
            subset_indices = None
            print("WARNING: No index file found, using all examples")

    # ── Load subset configs ─────────────────────────────────────
    configs = {}
    for name, cfg in SUBSET_CONFIGS.items():
        path = BASE_DIR / cfg["path"]
        if not path.exists():
            print(f"  SKIP: {name} — log not found: {path}")
            continue
        by_idx = load_log(path)
        if subset_indices is not None:
            indices = subset_indices & set(by_idx.keys())
        else:
            indices = set(by_idx.keys())
        acc = compute_accuracy(by_idx, indices, diff_map)
        configs[name] = {
            "description": cfg["description"],
            "flags": cfg["flags"],
            "dataset": f"500-subset ({len(indices)} matched)",
            "by_idx": by_idx,
            "indices": indices,
            **acc,
        }
        print(f"  {name}: {acc['correct']}/{acc['total']} = {acc['accuracy']:.4f}")

    # ── Load full configs ───────────────────────────────────────
    if args.include_full:
        for name, cfg in FULL_CONFIGS.items():
            path = BASE_DIR / cfg["path"]
            if not path.exists():
                print(f"  SKIP: {name} — log not found: {path}")
                continue
            by_idx = load_log(path)
            indices = set(by_idx.keys())
            acc = compute_accuracy(by_idx, indices, diff_map)
            configs[name] = {
                "description": cfg["description"],
                "flags": cfg["flags"],
                "dataset": f"full ({len(indices)} examples)",
                "by_idx": by_idx,
                "indices": indices,
                **acc,
            }
            print(f"  {name}: {acc['correct']}/{acc['total']} = {acc['accuracy']:.4f}")

    # ── Pairwise McNemar's tests ────────────────────────────────
    # Test each config against the baseline (C7_SQLGenOnly)
    baseline_name = "C7_SQLGenOnly"
    comparisons = {}

    if baseline_name in configs:
        base_data = configs[baseline_name]
        for name, cfg_data in configs.items():
            if name == baseline_name:
                continue
            shared = base_data["indices"] & cfg_data["indices"]
            if not shared:
                continue
            test = mcnemar_test(base_data["by_idx"], cfg_data["by_idx"], shared)
            delta_pct = round(
                (cfg_data["accuracy"] - base_data["accuracy"]) * 100, 2
            ) if cfg_data["accuracy"] is not None and base_data["accuracy"] is not None else None
            comparisons[f"{name}_vs_{baseline_name}"] = {
                "shared_examples": len(shared),
                "delta_pct": delta_pct,
                **test,
            }
            sig_str = "SIGNIFICANT" if test["significant"] else "not significant"
            print(f"  {name} vs {baseline_name}: delta={delta_pct:+.2f}%, "
                  f"chi2={test['chi2']:.4f}, {sig_str}")

    # Full-dataset comparison if both exist
    full_base = "C7_SQLGenOnly_full"
    full_comp = "C1_full_pipeline_full"
    if full_base in configs and full_comp in configs:
        base_data = configs[full_base]
        comp_data = configs[full_comp]
        shared = base_data["indices"] & comp_data["indices"]
        test = mcnemar_test(base_data["by_idx"], comp_data["by_idx"], shared)
        delta_pct = round(
            (comp_data["accuracy"] - base_data["accuracy"]) * 100, 2
        ) if comp_data["accuracy"] is not None and base_data["accuracy"] is not None else None
        comparisons[f"{full_comp}_vs_{full_base}"] = {
            "shared_examples": len(shared),
            "delta_pct": delta_pct,
            **test,
        }
        sig_str = "SIGNIFICANT" if test["significant"] else "not significant"
        print(f"  {full_comp} vs {full_base}: delta={delta_pct:+.2f}%, "
              f"chi2={test['chi2']:.4f}, {sig_str}")

    # ── Build output (strip by_idx to keep JSON small) ──────────
    output_configs = {}
    for name, cfg_data in configs.items():
        output_configs[name] = {
            k: v for k, v in cfg_data.items() if k not in ("by_idx", "indices")
        }

    summary = {
        "configs": output_configs,
        "comparisons_vs_baseline": comparisons,
    }

    # ── Print thesis-ready table ────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  ABLATION SUMMARY TABLE (for thesis)")
    print(f"{'=' * 80}")
    print(f"  {'Config':<25s} {'Acc':>12s} {'vs C7':>10s} {'chi2':>10s} {'Sig?':>8s}")
    print(f"  {'-' * 68}")

    # Order: offline_original, offline_no_step2, C7, noplanner, C1
    display_order = ["offline_original", "offline_no_step2", "C7_SQLGenOnly", "noplanner", "C1_full_pipeline"]
    for name in display_order:
        if name not in configs:
            continue
        c = configs[name]
        acc_str = f"{c['correct']}/{c['total']} ({c['accuracy']*100:.1f}%)"

        comp_key = f"{name}_vs_{baseline_name}"
        if name == baseline_name:
            vs_str = "baseline"
            chi2_str = "—"
            sig_str = "—"
        elif comp_key in comparisons:
            comp = comparisons[comp_key]
            vs_str = f"{comp['delta_pct']:+.1f}%"
            chi2_str = f"{comp['chi2']:.4f}"
            sig_str = "YES" if comp["significant"] else "no"
        else:
            vs_str = "—"
            chi2_str = "—"
            sig_str = "—"

        print(f"  {name:<25s} {acc_str:>12s} {vs_str:>10s} {chi2_str:>10s} {sig_str:>8s}")

    # Full-dataset section
    full_order = ["C7_SQLGenOnly_full", "C1_full_pipeline_full"]
    if any(n in configs for n in full_order):
        print(f"\n  {'— Full Dataset —':<25s}")
        for name in full_order:
            if name not in configs:
                continue
            c = configs[name]
            acc_str = f"{c['correct']}/{c['total']} ({c['accuracy']*100:.1f}%)"

            comp_key = f"{name}_vs_{full_base}"
            if name == full_base:
                vs_str = "baseline"
                chi2_str = "—"
                sig_str = "—"
            elif comp_key in comparisons:
                comp = comparisons[comp_key]
                vs_str = f"{comp['delta_pct']:+.1f}%"
                chi2_str = f"{comp['chi2']:.4f}"
                sig_str = "YES" if comp["significant"] else "no"
            else:
                vs_str = "—"
                chi2_str = "—"
                sig_str = "—"

            print(f"  {name:<25s} {acc_str:>12s} {vs_str:>10s} {chi2_str:>10s} {sig_str:>8s}")

    print()

    # ── Save JSON ───────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = BASE_DIR / "secondloop" / "logs" / "bird" / "ablation" / "ablation_summary.json"
    if not out_path.is_absolute():
        out_path = BASE_DIR / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved ablation summary to {out_path}")


if __name__ == "__main__":
    main()
