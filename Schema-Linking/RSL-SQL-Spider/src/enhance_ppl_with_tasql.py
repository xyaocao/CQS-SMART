"""
Enhance ppl_dev.json with column_info and column_meaning from spider_dev_samples.json.

This script:
1. Loads spider_dev_samples.json (rich column info)
2. Loads ppl_dev.json (schema-linked data)
3. Matches examples by db_id + question
4. Adds column_meaning and column_info to enhance schema text
5. Outputs ppl_dev_enhanced.json with richer schema information

The enhanced schema helps the reviewer catch issues like:
- Entity-attribute alignment (column_meaning shows what each column represents)
- Value format issues (sample_values show actual data format like 'F'/'T' vs 'Female'/'Male')
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import re


def load_json(path: Path) -> List[Dict]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_question(q: str) -> str:
    """Normalize question for matching."""
    return ' '.join(q.lower().split())


def build_tasql_index(tasql_data: List[Dict]) -> Dict[str, Dict]:
    """
    Build an index from spider_dev_samples.json for fast lookup.
    Key: (db_id, normalized_question)
    Value: the full spider_dev_samples.json record
    """
    index = {}
    for item in tasql_data:
        db_id = item.get('db_id', '')
        question = normalize_question(item.get('question', ''))
        key = (db_id, question)
        index[key] = item
    return index


def format_column_info(column_info: Dict[str, Dict], columns: List[str] = None) -> str:
    """
    Format column_info into a readable string.
    
    Args:
        column_info: Dict mapping table.column -> {type, sample_values}
        columns: Optional list of columns to include (schema-linked columns)
    
    Returns:
        Formatted string showing column types and sample values
    """
    if not column_info:
        return ""
    
    lines = []
    lines.append("### Column Info (type and sample values):")
    
    # Group by table
    tables = {}
    for col_key, info in column_info.items():
        if columns and col_key not in columns and f"`{col_key.split('.')[-1]}`" not in str(columns):
            # Filter to only schema-linked columns if provided
            # But be flexible with backtick variations
            continue
        
        parts = col_key.split('.')
        if len(parts) == 2:
            table, col = parts
        else:
            table, col = 'unknown', col_key
        
        if table not in tables:
            tables[table] = []
        
        col_type = info.get('type', 'unknown')
        samples = info.get('sample_values', [])
        sample_str = ', '.join(str(s) for s in samples[:5])  # Limit to 5 samples
        
        tables[table].append(f"  - {col} ({col_type}): [{sample_str}]")
    
    for table, cols in sorted(tables.items()):
        lines.append(f"# {table}:")
        lines.extend(cols)
    
    return '\n'.join(lines)


def format_column_meaning(column_meaning: Dict[str, str], columns: List[str] = None) -> str:
    """
    Format column_meaning into a readable string for entity-attribute alignment.
    
    Args:
        column_meaning: Dict mapping table.column -> human-readable description
        columns: Optional list of columns to include (schema-linked columns)
    
    Returns:
        Formatted string showing column semantics
    """
    if not column_meaning:
        return ""
    
    lines = []
    lines.append("### Column Meanings (what each column represents):")
    
    # Group by table
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


def create_enhanced_schema(ppl_item: Dict, tasql_item: Optional[Dict]) -> str:
    """
    Create an enhanced schema text combining PPL schema-linking with spider_dev_samples.json column info.
    
    This enhanced schema includes:
    1. Original DDL structure
    2. Column meanings (for entity-attribute alignment)
    3. Column info with sample values (for value format verification)
    4. Foreign keys
    """
    parts = []
    
    # 1. Original simplified DDL
    if 'simplified_ddl' in ppl_item:
        parts.append("### Database Schema:")
        parts.append(ppl_item['simplified_ddl'])
    
    # 2. DDL with sample data
    if 'ddl_data' in ppl_item:
        parts.append("\n### Schema with Sample Data:")
        parts.append(ppl_item['ddl_data'])
    
    # 3. Add column meanings from TA-SQL (CRITICAL for entity-attribute alignment)
    if tasql_item and 'column_meaning' in tasql_item:
        parts.append("\n" + format_column_meaning(
            tasql_item['column_meaning'],
            ppl_item.get('columns', [])
        ))
    
    # 4. Add column info with sample values from TA-SQL (for value format issues)
    if tasql_item and 'column_info' in tasql_item:
        parts.append("\n" + format_column_info(
            tasql_item['column_info'],
            ppl_item.get('columns', [])
        ))
    
    # 5. Foreign keys
    if 'foreign_key' in ppl_item:
        parts.append("\n### Foreign Keys:")
        parts.append(ppl_item['foreign_key'])
    
    # 6. Schema-linked tables and columns
    if 'tables' in ppl_item:
        parts.append(f"\n### tables: {ppl_item['tables']}")
    if 'columns' in ppl_item:
        parts.append(f"### columns: {ppl_item['columns']}")
    
    return '\n'.join(parts)


def enhance_ppl_data(ppl_data: List[Dict], tasql_index: Dict[str, Dict]) -> List[Dict]:
    """
    Enhance PPL data with spider_dev_samples.json column info and meanings.
    """
    enhanced = []
    matched = 0
    unmatched = 0
    
    for ppl_item in ppl_data:
        db_id = ppl_item.get('db', '')
        question = normalize_question(ppl_item.get('question', ''))
        key = (db_id, question)
        
        # Find matching TA-SQL item
        tasql_item = tasql_index.get(key)
        
        if tasql_item:
            matched += 1
        else:
            unmatched += 1
        
        # Create enhanced item
        enhanced_item = ppl_item.copy()
        
        # Add TA-SQL data
        if tasql_item:
            enhanced_item['column_info'] = tasql_item.get('column_info', {})
            enhanced_item['column_meaning'] = tasql_item.get('column_meaning', {})
            enhanced_item['primary_keys'] = tasql_item.get('primary_keys', {})
            # Foreign keys might be more detailed in TA-SQL
            if 'foreign_keys' in tasql_item:
                enhanced_item['foreign_keys_detailed'] = tasql_item['foreign_keys']
        
        # Create enhanced schema text
        enhanced_item['enhanced_schema'] = create_enhanced_schema(ppl_item, tasql_item)
        
        enhanced.append(enhanced_item)
    
    print(f"Matched: {matched}/{len(ppl_data)} ({100*matched/len(ppl_data):.1f}%)")
    print(f"Unmatched: {unmatched}")
    
    return enhanced


def main():
    # Paths
    script_dir = Path(__file__).parent
    
    # Input files
    tasql_path = Path("Data/spider_data/spider_dev_samples.json")
    ppl_path = script_dir / "information" / "ppl_dev.json"
    
    # Output file
    output_path = script_dir / "information" / "ppl_dev_enhanced.json"
    
    # Handle relative paths when running from different directories
    if not tasql_path.exists():
        # Try from project root
        project_root = script_dir.parent.parent.parent
        tasql_path = project_root / "Data/spider_data/spider_dev_samples.json"
    
    print(f"Loading spider_dev_samples.json data from: {tasql_path}")
    print(f"Loading PPL data from: {ppl_path}")
    
    # Load data
    tasql_data = load_json(tasql_path)
    ppl_data = load_json(ppl_path)
    
    print(f"spider_dev_samples.json samples: {len(tasql_data)}")
    print(f"PPL samples: {len(ppl_data)}")
    
    # Build index for fast lookup
    tasql_index = build_tasql_index(tasql_data)
    
    # Enhance PPL data
    enhanced_data = enhance_ppl_data(ppl_data, tasql_index)
    
    # Save enhanced data
    save_json(enhanced_data, output_path)
    print(f"\nSaved enhanced data to: {output_path}")
    
    # Show example
    print("\n" + "="*60)
    print("Example enhanced schema (first item):")
    print("="*60)
    if enhanced_data:
        print(enhanced_data[0].get('enhanced_schema', '')[:2000])


if __name__ == "__main__":
    main()

