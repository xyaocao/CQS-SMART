import json
from typing import Dict, Any, List, Tuple

def load_tables(tables_file: str) -> Dict[str, Any]:
    """Load table schemas from a JSON file."""
    with open(tables_file, 'r') as f:
        tables = json.load(f)
    return tables

def load_spider(tables_path: str) -> Dict[str, Any]:
    """Load Spider dataset table schemas."""
    return load_tables(tables_path)

def load_bird(tables_path: str) -> Dict[str, Any]:
    """Load BIRD dataset table schemas."""
    return load_tables(tables_path)

def load_schema(entries: List[Dict[str, Any]], db_id: str) -> str:
    """Extract schema text for a given database ID from dataset entries."""
    entry = next((e for e in entries if e.get('db_id') == db_id), None)
    if not entry:
        raise ValueError(f"db_id '{db_id}' not found in tables.")
    tables = entry.get('table_names_original') or entry.get('table_names') or []
    columns = entry.get('column_names_original') or entry.get('column_names') or []
    column_types = entry.get('column_types') or []
    primary_keys = entry.get('primary_keys') or []
    foreign_keys = entry.get('foreign_keys') or []
    
    column_idx_to_table_col: Dict[int, Tuple[str, str]] = {
        idx: (table_idx, column_name)
        for idx, (table_idx, column_name) in enumerate(columns)
        if table_idx != -1 and column_name
    }

    table_to_columns: Dict[int, List[Tuple[int, str, str]]] = {i: [] for i in range(len(tables))}
    for idx, (table_idx, column_name) in enumerate(columns):
        if table_idx == -1 or not column_name:
            continue
        column_type = column_types[idx] if idx < len(column_types) else None
        table_to_columns[table_idx].append((idx, column_name, column_type))
    
    fk_lookup: Dict[int, List[str]] = {}
    for src_idx, tgt_idx in foreign_keys:
        target = column_idx_to_table_col.get(tgt_idx)
        if not target:
            continue
        tgt_tbl_idx, tgt_col_name = target
        target_table = tables[tgt_tbl_idx] if tgt_tbl_idx < len(tables) else f"table_{tgt_tbl_idx}"
        fk_lookup.setdefault(src_idx, []).append(f"{target_table}.{tgt_col_name}")

    lines: List[str] = []
    for table_idx, table_name in enumerate(tables):
        lines.append(f"Table: {table_name}:")
        columns = table_to_columns.get(table_idx, [])
        if not columns:
            lines.append(" - (No columns found).")
            continue
        for col_idx, col_name, col_type in columns:
            markers: List[str] = []
            if col_idx in primary_keys:
                markers.append("PK")
            if col_idx in fk_lookup:
                refs = ",".join(fk_lookup[col_idx])
                markers.append(f"FK -> {refs}")
            marker_str = f" [{' '.join(markers)}]" if markers else ""
            type_str = f" ({col_type})" if col_type else ""
            lines.append(f" - {col_name}{type_str}{marker_str}")
    return "\n".join(lines)

def get_schema_from_spider(spider_tables: List[Dict[str, Any]], db_id: str) -> str:
    """Get schema text for a given db_id from Spider dataset."""
    return load_schema(spider_tables, db_id)

def get_schema_from_bird(bird_tables: List[Dict[str, Any]], db_id: str) -> str:
    """Get schema text for a given db_id from BIRD dataset."""
    return load_schema(bird_tables, db_id)