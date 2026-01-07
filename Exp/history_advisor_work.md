## History Advisor Workflow (current process)

1) **Initial advisor build (baseline ImprovedMAD run, no advisor)**
   - Source log: `Exp/logs/Qwen2.5-Coder/ImprovedMad_run1_log.json` (ImprovedMAD without history advisor).
   - Script: `Exp/extract_examples/build_history_advisor.py`.
   - Logic:
     - Load only rows with `exec_match == true`.
     - Filter to target DBs (high-error set).
     - Tag each SQL (join_fk, min_max, group_by, distinct, string_match, count).
     - Keep only “high-signal” exemplars (must have join_fk/min_max/group_by/distinct/string_match, or count paired with group_by/distinct).
     - Cap per-db (`--per-db-cap`, default 8) to balance coverage.
     - Write JSONL lines `{db_id, question, answer_sql, rationale, tags, source}`.
   - Command example:
     ```
     python Exp/extract_examples/build_history_advisor.py \
       --log Exp/logs/Qwen2.5-Coder/ImprovedMad_run1_log.json \
       --out Exp/history_advisor_q25.jsonl \
       --per-db-cap 8
     ```

2) **Find top mistake DBs**
   - Script: `Exp/mismatch_analyze/inspect_log.py`.
   - Run on the baseline log to get counts and top failing db_ids (e.g., top 15).
   - Command example:
     ```
     python Exp/mismatch_analyze/inspect_log.py \
       Exp/logs/Qwen2.5-Coder/ImprovedMad_run1_log.json --topn 15
     ```

3) **Run ImprovedMAD with advisor on top DBs (per-db slices)**
   - Use `evaluation.py` with `--loop_mode ImprovedMAD`, pointing the engine to the advisor file (rename to `Exp/history_advisor.jsonl` or pass the path in engine init).
   - Run small batches per high-error db index range (from the baseline log) to produce per-db test logs, e.g.:
     ```
     python Exp/evaluation.py --loop_mode ImprovedMAD \
       --start <range_start> --max_examples <range_len> \
       --log Exp/logs/Qwen2.5-Coder/test/improvedmad_new_<db>.json
     ```

4) **Diff old vs. new logs to find improved examples**
   - Script: `Exp/extract_history/compare_db_diff.py`.
   - Compares `exec_match` for a single db between old (run1) and new (per-db test) logs; prints improved indices and SQL.
   - Command example:
     ```
     python Exp/extract_history/compare_db_diff.py \
       --db world_1 \
       --old Exp/logs/Qwen2.5-Coder/ImprovedMad_run1_log.json \
       --new Exp/logs/Qwen2.5-Coder/test/improvedmad_new_world_1.json
     ```
   - Take the improved example_index entries (false→true) and append their `{db_id, question, answer_sql, rationale, tags, source}` to `Exp/history_advisor_q25.jsonl` to strengthen the advisor.


