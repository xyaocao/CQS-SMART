"""
Utility to load and convert RSL-SQL ppl_dev.json format for use in firstloop system.
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


def build_enhanced_schema_text(ppl_item: Dict[str, Any]) -> str:
    """
    Build enhanced schema text from ppl_dev.json item.
    Incorporates:
    - simplified_ddl (table structure)
    - ddl_data (sample data - first 3 rows)
    - foreign_key (foreign key relationships)
    - evidence (domain knowledge/definitions - BIRD dataset only, very important)
    - tables (schema linking results - relevant tables)
    - columns (schema linking results - relevant columns)
    - example (few-shot examples) - optional, can be used separately
    """
    parts = []
    
    # 1. Simplified DDL (table structure)
    if ppl_item.get("simplified_ddl"):
        parts.append(ppl_item["simplified_ddl"].strip())
    
    # 2. Sample data (first 3 rows) - helps understand data distribution
    if ppl_item.get("ddl_data"):
        parts.append("\n" + ppl_item["ddl_data"].strip())
    
    # 3. Foreign key relationships
    if ppl_item.get("foreign_key"):
        parts.append("\n" + ppl_item["foreign_key"].strip())
    
    # 4. Evidence/Definition (BIRD dataset) - VERY IMPORTANT domain knowledge
    # This provides critical information about how to interpret terms and calculate values
    # Format matches RSL-SQL style: "### definition: " (lowercase, as used in RSL-SQL prompts)
    if ppl_item.get("evidence"):
        evidence = ppl_item["evidence"].strip()
        if evidence:  # Only add if not empty
            parts.append("\n### definition: " + evidence)
    
    # 5. Schema linking results - highlight relevant tables and columns
    if ppl_item.get("tables") or ppl_item.get("columns"):
        schema_linking = []
        if ppl_item.get("tables"):
            schema_linking.append(f"### Relevant tables identified: {', '.join(ppl_item['tables'])}")
        if ppl_item.get("columns"):
            schema_linking.append(f"### Relevant columns identified: {', '.join(ppl_item['columns'])}")
        if schema_linking:
            parts.append("\n" + "\n".join(schema_linking))
    
    return "\n".join(parts)


def build_schema_text_with_examples(ppl_item: Dict[str, Any]) -> str:
    """
    Build schema text including few-shot examples.
    This is useful for SQL generation prompts.
    """
    base_schema = build_enhanced_schema_text(ppl_item)
    
    # Add few-shot examples if available
    if ppl_item.get("example"):
        base_schema += "\n\n" + ppl_item["example"].strip()
    
    return base_schema


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

