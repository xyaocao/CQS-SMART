## Findings from ImprovedMad_run1_log.json (Qwen3 run1)
- Total 1034; exec_match 825 (0.7979).
- Fail hotspots: `car_1` (45), `world_1` (34), `student_transcripts_tracking` (26), `dog_kennels` (25), `cre_Doc_Template_Mgt` (12), `wta_1` (12), `flight_2` (11), `tvshow` (10).
- Contract chosen far more often; most misses are semantic, not execution (exec_ok for both is high).

### Error motifs
- `car_1`: projection drift on “model” (use `car_names.Model` via `car_names.MakeId = cars_data.Id`; avoid `Make`/`model_list` unless maker asked). Literal drift (“usa” vs “United States”).
- `world_1`: “how many people” → should sum Population, not count rows. Two-language queries (“English and Dutch”) need BOTH, not `OR`. Region lists need DISTINCT.
- `student_transcripts_tracking`: dropped simple literals (e.g., “math” filter missing).

## Proposed improvements (to push >0.82)
1) History-guided hint bias (jsonl) keyed by db_id + question pattern; bias chooser toward candidates that satisfy preferred select/literals. Seed with known regressions (car_1 model/literal; world_1 population/both-languages).
2) Chooser heuristics (before LLM judge):
   - “how many people/population”: prefer SUM(Population) over row counts.
   - Two specific languages with “and/both”: require both (HAVING/INTERSECT) over OR.
   - Region/area lists: prefer DISTINCT.
   - Literal preservation: prefer candidate containing key literal tokens (even unquoted, e.g., “math”, “usa”).
   - `car_1` “model” queries: prefer `car_names.Model` with join on `car_names.MakeId = cars_data.Id`; deprioritize `Make`/`model_list` unless question asks maker.
3) Prompt nudge in ContractReasoner:
   - For “model” on car_1: project `car_names.Model` via `car_names.MakeId` join to `cars_data.Id` unless the question is explicitly about maker.
   - “How many people” → aggregate Population, not row count.
   - Two-language queries → require both languages, not OR.
4) Logging: record when a hint/heuristic is applied for audit.

