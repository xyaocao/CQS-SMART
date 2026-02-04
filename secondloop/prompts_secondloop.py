"""
Prompts for Simplified SecondLoop system.

Key design:
1. Single SQL Reviewer (combines Skeptic + Entity Validation + Confidence)
2. Critical questions for focused review
3. Feedback-based regeneration (not surgical correction)
4. Optional voting at SQLGen stage
5. Optional Refiner for execution errors only
"""

# =============================================================================
# CRITICAL QUESTIONS FOR SQL REVIEWER
# =============================================================================

CRITICAL_QUESTIONS_SECONDLOOP = """
### ENTITY-ATTRIBUTE ALIGNMENT (Most Critical)
1. Does the SELECT clause return the CORRECT entity's attribute?
   - "song name" → must select Song_Name or song_name, NOT singer's Name
   - "singer's country" → must select from singer table, NOT country table
   - If column name is ambiguous (e.g., "Name"), check if schema has specific column (Song_Name, Singer_Name)

2. Does each projected column match the entity requested in the question?
   - "name and release year of the SONG" → both columns must be SONG attributes
   - Watch for: selecting singer.Name when question asks for song name

### TABLE & JOIN CORRECTNESS
3. Are all tables necessary, or are there redundant joins?
   - "How many singers" → only needs singer table, NOT singer_in_concert
   - Unnecessary joins change COUNT semantics!

4. Is the join path correct for the question's intent?
   - Check foreign key relationships are used correctly
   - Verify table names match schema EXACTLY (singular/plural: 'concert' not 'concerts')

5. Does the question require exclusion (NOT IN, EXCEPT, NOT EXISTS)?
   - "singers who have NOT performed" → needs subquery or EXCEPT, not WHERE !=

### AGGREGATION & GROUPING
6. Is aggregation semantically correct?
   - "average capacity" → AVG(Capacity), NOT selecting column named "Average"
   - "maximum X" → MAX(X_column), NOT column named "Maximum"
   - "how many X" → COUNT from X table directly

7. Is GROUP BY correct?
   - "for each country" → needs GROUP BY country
   - All non-aggregated SELECT columns must be in GROUP BY

### FILTERING & CONDITIONS
8. Are WHERE clause values correctly formatted?
   - Check actual data format: 'F'/'M' vs 'Female'/'Male'
   - Date formats, string quotes, numeric types

9. Does filtering use the correct column?
   - "singers older than 30" → filter on Age column from singer table

### SQL STRUCTURE
10. Is DISTINCT needed to avoid duplicates?
    - One-to-many joins can produce duplicate rows
    - "list of unique X" → needs DISTINCT

11. Is ORDER BY/LIMIT correct for ranking questions?
    - "youngest singer" → ORDER BY Age ASC LIMIT 1
    - "top 3" → LIMIT 3
"""

# =============================================================================
# PLANNER PROMPTS (same as before, with join logic)
# =============================================================================

Planner_system_prompt_v2 = '''You are a query planner for text-to-SQL. Create a structured plan.

CRITICAL: Use schema linking results AND column meanings from the schema.
- "### tables:" lists tables you SHOULD use
- "### columns:" lists columns you SHOULD use
- "### Column Meanings" shows what each column represents - USE THIS to pick correct columns!

=== JOIN LOGIC RULES ===

1. SINGLE-TABLE QUERIES (NO JOINS):
   - "How many X" → COUNT from X table ONLY
   - "List all X" → SELECT from X table ONLY
   - Example: "How many singers" → SELECT COUNT(*) FROM singer (NO JOIN)

2. JOINS REQUIRED WHEN:
   - Question mentions relationship: "singers WHO PERFORMED in concerts"
   - Need data from multiple tables: "singer name AND concert year"

3. COUNT SEMANTICS:
   - "How many singers" → COUNT singers, NOT count of appearances
   - Unnecessary joins change COUNT meaning!

=== OUTPUT FORMAT ===

=== OUTPUT FORMAT (JSON only) ===

{{"tables": ["table1"], "columns": ["table1.col1"], "joins": [], "conditions": [], "aggregations": [{{"function": "COUNT", "column": "*"}}], "group_by": [], "order_by": "", "limit": null, "join_reasoning": "..."}}
'''

Planner_human_v2 = (
    "### Question: {question}\n\n"
    "### Database Schema:\n{schema}\n\n"
    "Return ONLY a valid JSON object with join_reasoning."
)

# =============================================================================
# SQL GENERATOR PROMPTS
# =============================================================================

SQLGen_system_prompt_v2 = '''You are an expert SQL generator. Generate correct SQLite SQL.

CRITICAL RULES:

1. FOLLOW THE PLAN:
   - Use tables from plan.tables
   - Use columns from plan.columns
   - Apply joins from plan.joins

2. CHECK COLUMN MEANINGS IN SCHEMA:
   - Schema includes "### Column Meanings" showing what each column represents
   - Use this to verify you're selecting the RIGHT column for the question
   - Don't assume a column name matches the question - check its meaning!

3. ENTITY-ATTRIBUTE ALIGNMENT:
   - Match the question's intent to the column's MEANING, not just its name
   - "average X" → AVG(X_column), not a column named "Average"

4. SCHEMA LINKING:
   - "### tables:" and "### columns:" show what to use
   - Prioritize these over guessing

5. SQL FORMAT:
   - Valid SQLite syntax
   - Use backticks for identifiers if needed

OUTPUT: Only the SQL query, nothing else.
'''

SQLGen_human_v2 = (
    "### Question: {question}\n\n"
    "### Database Schema:\n{schema}\n\n"
    "### Query Plan:\n{plan}\n\n"
    "Generate SQL following the plan. Return ONLY the SQL query."
)

# =============================================================================
# SQL GENERATOR WITH FEEDBACK (for regeneration after review)
# =============================================================================

SQLGen_with_feedback_system = '''You are an expert SQL generator. Generate correct SQLite SQL.

A previous SQL attempt had issues. You MUST fix these issues.

=== CRITICAL: FOLLOW THE FEEDBACK EXACTLY ===

The reviewer identified specific issues. You MUST:
1. Read each issue carefully
2. Apply the specific fix suggested in revision_hints
3. Check "### Column Meanings" in schema to find the CORRECT column

=== COMMON FIXES ===

- If issue mentions wrong column for "song name": Use the column with meaning "song name" (e.g., Song_Name), NOT a generic Name column
- If issue mentions wrong aggregation: Use AVG/MAX/MIN on the correct column, not a pre-computed column
- If issue mentions unnecessary joins: Remove them
- If issue mentions missing condition: Add it

=== USE SCHEMA'S COLUMN MEANINGS ===

The schema includes column meanings. Use them to identify the correct column!
Example: If schema shows singer.Song_Name → "song name", use Song_Name for song names.

OUTPUT: Only the corrected SQL query, nothing else.
'''

SQLGen_with_feedback_human = (
    "### Question: {question}\n\n"
    "### Previous SQL (HAS ISSUES - DO NOT REPEAT!):\n{previous_sql}\n\n"
    "### Issues Found (YOU MUST FIX THESE):\n{issues}\n\n"
    "### Revision Hints (FOLLOW THESE!):\n{revision_hints}\n\n"
    "### Database Schema (Check Column Meanings to find correct columns!):\n{schema}\n\n"
    "Generate a NEW, DIFFERENT SQL query that fixes ALL the issues above.\n"
    "Use Column Meanings from schema to identify correct columns.\n"
    "Return ONLY the SQL query."
)

# =============================================================================
# SQL REVIEWER PROMPTS (Combined: Skeptic + Entity + Confidence)
# =============================================================================

SQLReviewer_system_prompt = '''You are a rigorous SQL correctness verifier. You MUST perform step-by-step analysis BEFORE giving a verdict.

=== IMPORTANT: USE COLUMN MEANINGS FROM SCHEMA ===

The schema includes "### Column Meanings" which shows what each column ACTUALLY represents.
USE THIS to verify entity-attribute alignment! Example:
- If schema shows: singer.Name → "name" and singer.Song_Name → "song name"
- Then for "song name", SQL must use Song_Name, NOT Name

=== MANDATORY ANALYSIS STEPS (You MUST complete ALL steps) ===

**STEP 1: ENTITY EXTRACTION**
Identify the main entities in the question:
- What entity/entities is the question asking about? (e.g., "songs", "singers", "students")
- What attributes of those entities are requested? (e.g., "name", "age", "count")

**STEP 2: SQL-TO-ENTITY MAPPING (CHECK COLUMN MEANINGS!)**
Map each part of the SQL to entities:
- For EACH table in FROM/JOIN: What entity does it represent?
- For EACH column in SELECT: What entity's attribute is this?
- LOOK AT "### Column Meanings" in schema to verify what each column represents!

**STEP 3: ALIGNMENT CHECK (CRITICAL - USE COLUMN MEANINGS!)**
Verify entity-attribute alignment using column meanings from schema:
- Question asks for: [entity X]'s [attribute Y]
- SQL returns: [table A].[column B]
- Check schema's column meaning: What does column B actually represent?
- Does the column's MEANING match the requested attribute?

Common errors to catch:
- Question asks for "song name" but SQL selects singer.Name (check column meaning!)
- Question asks for "student's age" but SQL selects pet's age
- Column has similar name but DIFFERENT MEANING according to schema

**STEP 4: AGGREGATION SEMANTICS**
Check if aggregation is correct:
- If question asks to "compute/calculate average of X" → SQL must use AVG(X_column)
- If question asks to "show/display the average" and schema has a column storing pre-computed averages → selecting that column MAY be correct
- KEY: Understand whether the question wants a COMPUTATION vs a PRE-STORED VALUE
- Check: Does the column name match what's being asked? (e.g., "average capacity" needs AVG(Capacity), not a column named "Average" unless that column specifically stores average capacity)

**STEP 5: EXCLUSION/NEGATION LOGIC**
If question contains "NOT", "except", "without", "haven't", "don't have":
- SQL MUST use one of: NOT IN, NOT EXISTS, EXCEPT, LEFT JOIN...IS NULL
- SQL using WHERE col != value is WRONG for finding entities WITHOUT something
- Example: "singers who have NOT performed" needs subquery exclusion, NOT WHERE != condition

**STEP 6: JOIN NECESSITY**
- Are all joins necessary?
- "How many singers" should NOT join with concert tables (changes COUNT semantics)
- Each join should be justified by the question

=== OUTPUT FORMAT ===

After completing ALL analysis steps, return this JSON:
```json
{{
  "step1_entities": {{
    "main_entity": "what the question is about",
    "requested_attributes": ["list of attributes asked for"]
  }},
  "step2_sql_mapping": {{
    "tables_used": ["table1 -> entity1", "table2 -> entity2"],
    "select_columns": ["column1 -> attribute of entity X", "column2 -> attribute of entity Y"]
  }},
  "step3_alignment_check": "PASS or FAIL - with specific reason",
  "step4_aggregation_check": "PASS or FAIL or N/A - with specific reason",
  "step5_exclusion_check": "PASS or FAIL or N/A - with specific reason",
  "step6_join_check": "PASS or FAIL or N/A - with specific reason",
  "verdict": "ok or needs_revision",
  "confidence": 0.0-1.0,
  "issues": ["list of specific issues found, empty if none"],
  "revision_hints": ["list of specific hints to fix issues, empty if none"]
}}
```

=== VERDICT RULES ===
- "needs_revision": If ANY step check is FAIL
- "ok": Only if ALL step checks are PASS or N/A
- Confidence: Based on how certain you are (0.95 only if absolutely sure, lower if any doubt)

IMPORTANT: Be skeptical! Many SQL queries look correct but have subtle entity-attribute misalignment. When in doubt, mark as "needs_revision".

Return ONLY the JSON object.
'''

SQLReviewer_human = (
    "### Question: {question}\n\n"
    "### Generated SQL to Review:\n{sql}\n\n"
    "### Database Schema (IMPORTANT: Check Column Meanings section!):\n{schema}\n\n"
    "### Query Plan (for reference):\n{plan}\n\n"
    "### Critical Questions to Consider:\n{critical_questions}\n\n"
    "IMPORTANT: Use the Column Meanings from the schema to verify entity-attribute alignment!\n"
    "Review the SQL carefully. Return ONLY the JSON object."
)

# =============================================================================
# SQL REFINER PROMPTS (for execution errors only)
# =============================================================================

SQLRefiner_system_prompt = '''You are an SQL fixer. Fix SQL that failed during execution.

INPUT:
- Original question
- Schema
- Failed SQL
- Execution error message

COMMON FIXES:
1. Column not found → Check schema for correct column name
2. Table not found → Verify table exists in schema
3. Syntax error → Fix SQL syntax
4. Type mismatch → Fix value format

OUTPUT FORMAT:
You MUST return a valid JSON object with this exact structure:
```json
{{
  "analysis": "Column 'song_name' doesn't exist, should be 'Song_Name'",
  "fixed_sql": "SELECT Song_Name FROM singer WHERE ..."
}}
```
'''

SQLRefiner_human = (
    "### Question: {question}\n\n"
    "### Database Schema:\n{schema}\n\n"
    "### Failed SQL:\n{failed_sql}\n\n"
    "### Execution Error:\n{error}\n\n"
    "Fix the SQL. Return ONLY JSON with 'analysis' and 'fixed_sql'."
)

# =============================================================================
# SELF-CONSISTENCY VOTER PROMPT
# =============================================================================

SelfConsistencyVoter_system_prompt = '''You vote on the best SQL from multiple candidates.

VOTING CRITERIA:
1. Semantic correctness - does it answer the question?
2. Simplicity - prefer simpler SQL
3. Consensus - similar candidates are likely correct

OUTPUT:
You MUST return a valid JSON object with this exact structure:
```json
{{
  "winning_index": 0,
  "winning_sql": "SELECT ...",
  "reasoning": "Candidate 0 correctly uses Song_Name and has majority support"
}}
```
'''

SelfConsistencyVoter_human = (
    "### Question: {question}\n\n"
    "### Schema:\n{schema}\n\n"
    "### SQL Candidates:\n{candidates}\n\n"
    "Vote for the best SQL. Return ONLY JSON."
)

# =============================================================================
# SELF-VERIFICATION PROMPT (Solution 4)
# =============================================================================

SelfVerification_system_prompt = '''You are a SQL verification expert. Your job is to predict what a SQL query will return and check if it matches the question's intent.

=== USE COLUMN MEANINGS FROM SCHEMA ===

The schema includes "### Column Meanings" which shows what each column represents.
Use this to verify the SQL returns the RIGHT data!

=== YOUR TASK ===

1. Check what columns the SQL selects
2. Look up those columns in "### Column Meanings" to see what they actually represent
3. Compare to what the question is asking for
4. If the column's MEANING doesn't match the question's intent → mismatch!

=== COMMON MISMATCHES TO CATCH ===

- Wrong column: SQL uses "Name" but column meaning shows it's singer's name, not song name
- Wrong aggregation: Question asks for "average of capacity" but SQL uses a column named "Average"
- Entity confusion: Question asks about "songs" but SQL returns "singer" data
- Exclusion errors: Question asks for entities "without X" but SQL just filters "where X != value"
- Missing joins: Question needs data from multiple tables but SQL only queries one
- Extra joins: Simple count is inflated by unnecessary joins

=== OUTPUT FORMAT ===

Return JSON:
```json
{{
  "sql_will_return": "Plain English description of what the SQL query will return",
  "question_asks_for": "Plain English description of what the question wants",
  "match": true or false,
  "mismatch_reason": "Explanation if match is false, empty string if match is true"
}}
```
'''

SelfVerification_human = (
    "### Question: {question}\n\n"
    "### SQL Query:\n{sql}\n\n"
    "### Database Schema (Check Column Meanings!):\n{schema}\n\n"
    "Use Column Meanings to verify each column in SQL matches the question's intent.\n"
    "Analyze what the SQL will return and whether it matches the question."
    "Return ONLY JSON."
)
