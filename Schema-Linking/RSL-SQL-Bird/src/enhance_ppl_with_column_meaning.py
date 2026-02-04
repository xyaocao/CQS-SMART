"""
Enhance BIRD ppl_dev.json with column_meaning from column_meaning.json.

This script:
1. Loads column_meaning.json (rich column descriptions)
2. Loads ppl_dev.json (schema-linked data)
3. For each example, extracts relevant column meanings based on db_id and schema-linked columns
4. Outputs ppl_dev_enhanced.json with column meanings embedded

This makes BIRD data consistent with Spider's enhanced format.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_column_meaning_index(column_meaning_data: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Build an index for fast lookup by db_id.
    
    Input format: "db_id|table|column": "description"
    Output format: {db_id: {"table.column": "description", ...}, ...}
    """
    index = {}
    for key, description in column_meaning_data.items():
        parts = key.split('|')
        if len(parts) == 3:
            db_id, table, column = parts
            if db_id not in index:
                index[db_id] = {}
            # Store with table.column format for consistency with Spider
            index[db_id][f"{table}.{column}"] = description
    return index


def get_column_meanings_for_example(
    ppl_item: Dict[str, Any],
    cm_index: Dict[str, Dict[str, str]]
) -> Dict[str, str]:
    """
    Get column meanings relevant to a specific example.
    
    Args:
        ppl_item: PPL example with 'db', 'tables', 'columns'
        cm_index: Index of column meanings by db_id
    
    Returns:
        Dict mapping table.column -> description for relevant columns
    """
    db_id = ppl_item.get('db', '')
    columns = ppl_item.get('columns', [])
    
    if db_id not in cm_index:
        return {}
    
    db_meanings = cm_index[db_id]
    relevant_meanings = {}
    
    # Normalize column names (remove backticks)
    for col in columns:
        # Handle formats like: table.`column` or `table`.`column`
        normalized = col.replace('`', '').lower()
        
        # Try to find a match in the meanings
        for meaning_key, description in db_meanings.items():
            meaning_key_lower = meaning_key.lower()
            if normalized == meaning_key_lower or normalized.endswith('.' + meaning_key_lower.split('.')[-1]):
                # Use original case from column_meaning.json
                relevant_meanings[meaning_key] = description
                break
    
    # Also include ALL column meanings for this database (useful for context)
    # But mark them separately so we know which were schema-linked
    return relevant_meanings


def get_all_db_column_meanings(
    db_id: str,
    cm_index: Dict[str, Dict[str, str]]
) -> Dict[str, str]:
    """Get all column meanings for a database."""
    return cm_index.get(db_id, {})


def format_column_meaning_text(column_meaning: Dict[str, str]) -> str:
    """
    Format column meanings into readable text for schema.
    Groups by table for better organization.
    """
    if not column_meaning:
        return ""
    
    lines = ["### The meaning of every column:"]
    lines.append("#")
    
    # Group by table
    tables = {}
    for col_key, description in column_meaning.items():
        parts = col_key.split('.')
        if len(parts) == 2:
            table, col = parts
        else:
            table, col = 'unknown', col_key
        
        if table not in tables:
            tables[table] = []
        
        # Clean up description - remove leading # if present
        desc = description.strip()
        if desc.startswith('#'):
            desc = desc[1:].strip()
        
        # Truncate very long descriptions
        if len(desc) > 200:
            desc = desc[:200] + "..."
        
        tables[table].append(f"# {col}: {desc}")
    
    for table, cols in sorted(tables.items()):
        lines.append(f"# [{table}]")
        lines.extend(cols)
    
    lines.append("#")
    return '\n'.join(lines)


def enhance_ppl_data(
    ppl_data: List[Dict],
    cm_index: Dict[str, Dict[str, str]]
) -> List[Dict]:
    """
    Enhance PPL data with column meanings.
    """
    enhanced = []
    
    for ppl_item in ppl_data:
        db_id = ppl_item.get('db', '')
        
        # Create enhanced item
        enhanced_item = ppl_item.copy()
        
        # Get all column meanings for this database
        all_meanings = get_all_db_column_meanings(db_id, cm_index)
        
        # Get relevant meanings for schema-linked columns
        relevant_meanings = get_column_meanings_for_example(ppl_item, cm_index)
        
        # Add to enhanced item
        enhanced_item['column_meaning'] = all_meanings
        enhanced_item['column_meaning_linked'] = relevant_meanings
        
        # Also add formatted text for direct use in prompts
        enhanced_item['column_meaning_text'] = format_column_meaning_text(all_meanings)
        
        enhanced.append(enhanced_item)
    
    # Print stats
    dbs_with_meanings = len(set(item.get('db', '') for item in enhanced if item.get('column_meaning')))
    total_dbs = len(set(item.get('db', '') for item in ppl_data))
    print(f"Databases with column meanings: {dbs_with_meanings}/{total_dbs}")
    
    return enhanced


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    # Input files
    column_meaning_path = project_root / "Data/BIRD/dev/column_meaning.json"
    ppl_path = script_dir / "information" / "ppl_dev.json"
    
    # Output file
    output_path = script_dir / "information" / "ppl_dev_enhanced.json"
    
    print(f"Loading column_meaning from: {column_meaning_path}")
    print(f"Loading PPL data from: {ppl_path}")
    
    # Load data
    column_meaning_data = load_json(column_meaning_path)
    ppl_data = load_json(ppl_path)
    
    print(f"Column meanings: {len(column_meaning_data)} entries")
    print(f"PPL samples: {len(ppl_data)}")
    
    # Build index for fast lookup
    cm_index = build_column_meaning_index(column_meaning_data)
    print(f"Databases in column_meaning: {len(cm_index)}")
    
    # Enhance PPL data
    enhanced_data = enhance_ppl_data(ppl_data, cm_index)
    
    # Save enhanced data
    save_json(enhanced_data, output_path)
    print(f"\nSaved enhanced data to: {output_path}")
    
    # Show example
    print("\n" + "="*60)
    print("Example enhanced item (first):")
    print("="*60)
    if enhanced_data:
        example = enhanced_data[0]
        print(f"Database: {example.get('db', 'N/A')}")
        print(f"Question: {example.get('question', 'N/A')[:100]}...")
        print(f"Column meanings count: {len(example.get('column_meaning', {}))}")
        print(f"\nColumn meaning text preview:")
        print(example.get('column_meaning_text', '')[:1000])


if __name__ == "__main__":
    main()

