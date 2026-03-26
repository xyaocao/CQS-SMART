"""
Generate a stratified 500-example subset from BIRD dev.json.

Distribution strategy (over-sample challenging):
  - challenging: ALL 145
  - moderate:    200 (random sample)
  - simple:      155 (random sample)
  Total: 500

Output: bird_dev_500.json  (sorted list of original indices)

Usage:
    python secondloop/tools/generate_subset.py
"""
import json
import random
from pathlib import Path
from collections import Counter

SEED = 42
TARGET_TOTAL = 500


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent  # project root
    dev_path = base_dir / "Data" / "BIRD" / "dev" / "dev.json"

    if not dev_path.exists():
        print(f"ERROR: {dev_path} not found")
        return

    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")

    # Group indices by difficulty
    buckets: dict[str, list[int]] = {}
    for i, item in enumerate(data):
        diff = item.get("difficulty", "unknown").lower()
        buckets.setdefault(diff, []).append(i)

    print("\nOriginal distribution:")
    for diff in sorted(buckets):
        print(f"  {diff:15s}: {len(buckets[diff]):4d}")

    # Allocation: all challenging, 200 moderate, rest simple
    random.seed(SEED)

    challenging = buckets.get("challenging", [])
    moderate = buckets.get("moderate", [])
    simple = buckets.get("simple", [])

    n_challenging = len(challenging)  # take all
    n_moderate = min(200, len(moderate))
    n_simple = TARGET_TOTAL - n_challenging - n_moderate

    if n_simple > len(simple):
        print(f"WARNING: need {n_simple} simple but only {len(simple)} available, adjusting")
        n_simple = len(simple)

    selected = (
        challenging
        + sorted(random.sample(moderate, n_moderate))
        + sorted(random.sample(simple, n_simple))
    )
    selected.sort()

    # Verify distribution
    dist = Counter()
    for idx in selected:
        diff = data[idx].get("difficulty", "unknown").lower()
        dist[diff] += 1

    print(f"\nSubset distribution ({len(selected)} total):")
    for diff in sorted(dist):
        print(f"  {diff:15s}: {dist[diff]:4d}")

    # Write output
    out_path = Path(__file__).resolve().parent / "bird_dev_500.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    print(f"\nSaved {len(selected)} indices to {out_path}")


if __name__ == "__main__":
    main()
