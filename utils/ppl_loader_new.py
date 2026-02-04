"""
Enhanced utility to load and convert RSL-SQL ppl_dev.json format for use in firstloop system.

Key improvements over original ppl_loader.py:
1. build_enhanced_schema_text() now includes few-shot examples by default (critical for accuracy)
2. Schema linking results are formatted more prominently as hard constraints
3. Added build_rsl_style_prompt() for direct SQL generation matching RSL-SQL's proven approach
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_ppl_data(ppl_json_path: str) -> List[Dict[str, Any]]:
    """Load ppl_dev.json file."""
    with open(ppl_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_ppl_to_examples(ppl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert ppl_dev.json format to standard example format.
    Changes 'db' to 'db_id' and preserves all other fields.
    """
    examples = []
    for item in ppl_data:
        example = item.copy()
        # Convert 'db' to 'db_id' for consistency with firstloop system
        if "db" in example:
            example["db_id"] = example.pop("db")
        examples.append(example)
    return examples


def format_column_meaning(column_meaning: Dict[str, str]) -> str:
    """Format column_meaning dict into readable text."""
    if not column_meaning:
        return ""
    
    lines = ["### Column Meanings (what each column represents):"]
    tables = {}
    
    for col_key, meaning in column_meaning.items():
        parts = col_key.split('.')
        if len(parts) == 2:
            table, col = parts
        else:
            table, col = 'unknown', col_key
        
        if table not in tables:
            tables[table] = []
        tables[table].append(f"  - {col} → \"{meaning}\"")
    
    for table, cols in sorted(tables.items()):
        lines.append(f"# {table}:")
        lines.extend(cols)
    
    return '\n'.join(lines)


def format_column_info(column_info: Dict[str, Dict]) -> str:
    """Format column_info dict with types and sample values."""
    if not column_info:
        return ""
    
    lines = ["### Column Types and Sample Values:"]
    tables = {}
    
    for col_key, info in column_info.items():
        parts = col_key.split('.')
        if len(parts) == 2:
            table, col = parts
        else:
            table, col = 'unknown', col_key
        
        if table not in tables:
            tables[table] = []
        
        col_type = info.get('type', 'unknown')
        samples = info.get('sample_values', [])
        sample_str = ', '.join(str(s) for s in samples[:5])
        tables[table].append(f"  - {col} ({col_type}): [{sample_str}]")
    
    for table, cols in sorted(tables.items()):
        lines.append(f"# {table}:")
        lines.extend(cols)
    
    return '\n'.join(lines)


def build_enhanced_schema_text(ppl_item: Dict[str, Any], include_examples: bool = True) -> str:
    """
    Build enhanced schema text from ppl_dev.json item.

    IMPORTANT: Now includes few-shot examples by default (include_examples=True).
    RSL-SQL achieves 82% accuracy partly because it uses few-shot examples.

    Incorporates:
    - simplified_ddl (table structure)
    - ddl_data (sample data - first 3 rows)
    - foreign_key (foreign key relationships)
    - column_meaning (what each column represents - from TA-SQL enhancement)
    - column_info (column types and sample values - from TA-SQL enhancement)
    - evidence (domain knowledge/definitions - BIRD dataset only, very important)
    - tables (schema linking results - relevant tables) - AS HARD CONSTRAINTS
    - columns (schema linking results - relevant columns) - AS HARD CONSTRAINTS
    - example (few-shot examples) - NOW INCLUDED BY DEFAULT
    """
    parts = []

    # 1. Simplified DDL (table structure)
    if ppl_item.get("simplified_ddl"):
        parts.append("### Sqlite SQL tables, with their properties:")
        parts.append(ppl_item["simplified_ddl"].strip())

    # 2. Sample data (first 3 rows) - helps understand data distribution and value formats
    if ppl_item.get("ddl_data"):
        parts.append("\n### Here are some data information about database references.")
        parts.append(ppl_item["ddl_data"].strip())

    # 3. Foreign key relationships
    if ppl_item.get("foreign_key"):
        parts.append("\n### Foreign key information of Sqlite SQL tables, used for table joins:")
        parts.append(ppl_item["foreign_key"].strip())

    # 4. Column Meanings - CRITICAL FOR ENTITY-ATTRIBUTE ALIGNMENT
    # This shows what each column semantically represents
    # Handle both Spider format (column_meaning dict) and BIRD format (column_meaning_text string)
    if ppl_item.get("column_meaning_text"):
        # BIRD format: pre-formatted text from column_meaning.json
        parts.append("\n" + ppl_item["column_meaning_text"])
    elif ppl_item.get("column_meaning"):
        # Spider format: dict from TA-SQL enhancement
        meaning_text = format_column_meaning(ppl_item["column_meaning"])
        if meaning_text:
            parts.append("\n" + meaning_text)

    # 5. Column Info with Sample Values (from TA-SQL enhancement for Spider) - FOR VALUE FORMAT VERIFICATION
    # This shows column types and actual sample values like 'F'/'T' vs 'Female'/'Male'
    # Note: BIRD already includes this info in ddl_data and column_meaning_text
    if ppl_item.get("column_info"):
        info_text = format_column_info(ppl_item["column_info"])
        if info_text:
            parts.append("\n" + info_text)

    # 6. Evidence/Definition (BIRD dataset) - VERY IMPORTANT domain knowledge
    # This provides critical information about how to interpret terms and calculate values
    if ppl_item.get("evidence"):
        evidence = ppl_item["evidence"].strip()
        if evidence:  # Only add if not empty
            parts.append("\n### definition: " + evidence)

    # 7. Schema linking results - THESE ARE HARD CONSTRAINTS, NOT SUGGESTIONS
    # Format matches RSL-SQL style for direct compatibility
    if ppl_item.get("tables") or ppl_item.get("columns"):
        parts.append("\n### Schema Linking Results (USE THESE TABLES AND COLUMNS):")
        if ppl_item.get("tables"):
            tables_str = str(ppl_item['tables'])
            parts.append(f"### tables: {tables_str}")
        if ppl_item.get("columns"):
            columns_str = str(ppl_item['columns'])
            parts.append(f"### columns: {columns_str}")

    # 7.5. Information augmentation from RSL-SQL-Bird step 2 (if available)
    # These provide hints about SQL keywords and conditions to use
    if ppl_item.get("sql_keywords"):
        keywords_str = str(ppl_item['sql_keywords'])
        parts.append(f"\n### sql_keywords: {keywords_str}")
    if ppl_item.get("conditions"):
        conditions_str = str(ppl_item['conditions'])
        parts.append(f"### conditions: {conditions_str}")

    # 8. Few-shot examples - CRITICAL FOR ACCURACY (now included by default)
    if include_examples and ppl_item.get("example"):
        parts.append("\n" + ppl_item["example"].strip())

    # 9. If enhanced_schema is pre-built (from enhance_ppl_with_tasql.py), use it
    # This is an alternative path when the enhanced file is already generated
    if ppl_item.get("enhanced_schema") and not ppl_item.get("column_meaning"):
        return ppl_item["enhanced_schema"]

    return "\n".join(parts)


def build_schema_text_without_examples(ppl_item: Dict[str, Any]) -> str:
    """
    Build schema text WITHOUT few-shot examples.
    Use this only when you want to handle examples separately.
    """
    return build_enhanced_schema_text(ppl_item, include_examples=False)


def build_rsl_style_prompt(ppl_item: Dict[str, Any], question: str) -> str:
    """
    Build a prompt that matches RSL-SQL's proven SQL generation approach.

    This creates a prompt structure identical to RSL-SQL's step_1_preliminary_sql.py
    which achieves ~82% execution accuracy on Spider.

    Structure:
    1. Few-shot examples (if available)
    2. SQL generation instruction
    3. Schema info (DDL, sample data, foreign keys)
    4. Schema linking results (tables, columns)
    5. Definition/evidence (if available)
    6. Question
    """
    prompt_parts = []

    # 1. Few-shot examples FIRST (RSL-SQL puts these at the beginning)
    if ppl_item.get("example"):
        prompt_parts.append(ppl_item["example"].strip())

    # 2. Instruction (matches RSL-SQL)
    prompt_parts.append("### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.")

    # 3. Build table_info (matches RSL-SQL's table_info_construct)
    table_info_parts = []

    if ppl_item.get("simplified_ddl"):
        table_info_parts.append("### Sqlite SQL tables, with their properties:")
        table_info_parts.append(ppl_item["simplified_ddl"].strip())

    if ppl_item.get("ddl_data"):
        table_info_parts.append("\n### Here are some data information about database references.")
        table_info_parts.append(ppl_item["ddl_data"].strip())

    if ppl_item.get("foreign_key"):
        table_info_parts.append("\n### Foreign key information of Sqlite SQL tables, used for table joins:")
        table_info_parts.append(ppl_item["foreign_key"].strip())

    # 4. Schema linking results (matches RSL-SQL format exactly)
    if ppl_item.get("tables"):
        table_info_parts.append(f"\n### tables: {ppl_item['tables']}")
    if ppl_item.get("columns"):
        table_info_parts.append(f"### columns: {ppl_item['columns']}")

    prompt_parts.append("\n".join(table_info_parts))

    # 5. Definition/evidence (for BIRD dataset)
    if ppl_item.get("evidence"):
        evidence = ppl_item["evidence"].strip()
        if evidence:
            prompt_parts.append("### definition: " + evidence)

    # 6. Question
    prompt_parts.append(f"### Question: {question}")

    return "\n\n".join(prompt_parts)


def get_ppl_example_by_index(ppl_data: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    """Get a specific example by index from ppl_data."""
    if 0 <= index < len(ppl_data):
        return ppl_data[index]
    return None


def get_ppl_example_by_db_and_question(
    ppl_data: List[Dict[str, Any]],
    db_id: str,
    question: str
) -> Optional[Dict[str, Any]]:
    """Find a ppl example by db_id and question (fuzzy match)."""
    for item in ppl_data:
        item_db = item.get("db") or item.get("db_id", "")
        item_question = item.get("question", "").strip()
        if item_db == db_id and item_question.strip() == question.strip():
            return item
    return None


def extract_schema_linking_info(ppl_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract schema linking information as a structured dict.
    Useful for validation and debugging.
    """
    return {
        "tables": ppl_item.get("tables", []),
        "columns": ppl_item.get("columns", []),
        "has_evidence": bool(ppl_item.get("evidence", "").strip()),
        "has_examples": bool(ppl_item.get("example", "").strip()),
    }


def validate_plan_against_schema_linking(plan: Dict[str, Any], ppl_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a planner's output against schema linking results.
    Returns a dict with validation results and any discrepancies.

    This helps catch cases where the planner deviates from schema linking.
    """
    sl_tables = set(ppl_item.get("tables", []))
    sl_columns = set(ppl_item.get("columns", []))

    plan_tables = set(plan.get("tables", []))
    plan_columns = set()
    for col in plan.get("columns", []):
        if isinstance(col, str):
            plan_columns.add(col)

    # Find discrepancies
    extra_tables = plan_tables - sl_tables
    missing_tables = sl_tables - plan_tables

    # Normalize column names for comparison (remove backticks, handle table.column format)
    def normalize_col(col):
        return col.replace("`", "").lower().strip()

    sl_cols_normalized = {normalize_col(c) for c in sl_columns}
    plan_cols_normalized = {normalize_col(c) for c in plan_columns}

    return {
        "valid": len(extra_tables) == 0,  # No extra tables beyond schema linking
        "extra_tables": list(extra_tables),
        "missing_tables": list(missing_tables),
        "sl_tables": list(sl_tables),
        "plan_tables": list(plan_tables),
        "warnings": [
            f"Plan includes table '{t}' not in schema linking results"
            for t in extra_tables
        ]
    }
