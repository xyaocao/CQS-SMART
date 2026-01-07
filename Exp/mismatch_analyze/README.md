## Mismatch Analysis Toolkit

Contents:
- `inspect_log.py`: CLI to summarize ImprovedMAD runs.
- `conclusions.md`: Findings and improvement plan from ImprovedMad_run1_log.json (acc 0.7979).

Usage:
```bash
python Exp/mismatch_analyze/inspect_log.py Exp/logs/Qwen3/ImprovedMad_run1_log.json --topn 10 --sample-db car_1 --sample-db world_1
```

What it prints:
- Total items and success counts (exec_match / variants).
- Top failing db_ids.
- Chosen candidate distribution (contract vs oneshot) for successes and failures.
- Optional sampled failures per db_id (question + truncated candidate SQLs).

