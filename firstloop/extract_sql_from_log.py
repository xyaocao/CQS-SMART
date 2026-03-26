"""
Extract final_sql from firstloop log JSON file and save to a text file (one SQL per line).
python Exp/extract_sql_from_log.py --log_file <log.json> --output_file <extracted_sql/sql_log_initial.txt>
python Exp/extract_sql_from_log.py --log_file secondloop/logs/bird/run_log.json --output_file results/extracted_sql/BIRD/run1_log.txt
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_sql_from_log(log_file: str, output_file: str):
    """
    Extract final_sql from firstloop log JSON file.
    
    Args:
        log_file: Path to firstloop log JSON file
        output_file: Path to output text file (one SQL per line)
    """
    log_path = Path(log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    # Read log file (can be JSON array or JSONL)
    entries = []
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            print(f"Warning: Log file is empty: {log_file}")
            return
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                entries = parsed
            elif isinstance(parsed, dict):
                # Single entry or wrapped in 'entries'
                if "entries" in parsed and isinstance(parsed["entries"], list):
                    entries = parsed["entries"]
                else:
                    entries = [parsed]
        except json.JSONDecodeError:
            # Try JSONL format (one JSON object per line)
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    
    if not entries:
        print(f"Warning: No entries found in log file: {log_file}")
        return
    
    # Extract final_sql from each entry
    sql_list = []
    missing_count = 0
    
    for entry in tqdm(entries, desc="Extracting SQL"):
        # Try different possible field names
        final_sql = entry.get("final_sql") or entry.get("sql") or entry.get("generated_sql")

        # If not present at top-level, some logs embed SQLs under an `sql_tracking` field
        # which may be a list of SQLs or a dict. Try to extract the most recent SQL from it.
        if not final_sql:
            sql_tracking = entry.get("sql_tracking") or entry.get("sql_tracking_set")
            candidate = None
            if sql_tracking:
                # string -> use directly
                if isinstance(sql_tracking, str):
                    candidate = sql_tracking
                elif isinstance(sql_tracking, dict):
                    # Common patterns: {'final_sql': '...', ...} or values are lists per round
                    for k in ("final_sql", "sql", "generated_sql"):
                        v = sql_tracking.get(k)
                        if v:
                            if isinstance(v, list):
                                candidate = v[-1]
                            else:
                                candidate = v
                            break
                    # If still not found, try last string-ish value in the dict
                    if not candidate:
                        for v in reversed(list(sql_tracking.values())):
                            if isinstance(v, str) and v.strip():
                                candidate = v
                                break
                            if isinstance(v, list) and v:
                                for item in reversed(v):
                                    if isinstance(item, str) and item.strip():
                                        candidate = item
                                        break
                                if candidate:
                                    break
                elif isinstance(sql_tracking, list):
                    # Find the most recent string or dict-containing SQL
                    for item in reversed(sql_tracking):
                        if isinstance(item, str) and item.strip():
                            candidate = item
                            break
                        if isinstance(item, dict):
                            for k in ("final_sql", "sql", "generated_sql"):
                                v = item.get(k)
                                if v:
                                    if isinstance(v, list):
                                        candidate = v[-1]
                                    else:
                                        candidate = v
                                    break
                            if candidate:
                                break
            if candidate:
                final_sql = candidate

        if final_sql:
            # If final_sql is a list (multiple rounds), pick the last non-empty element
            if isinstance(final_sql, list):
                for item in reversed(final_sql):
                    if isinstance(item, str) and item.strip():
                        final_sql = item
                        break
                else:
                    final_sql = ""

        if final_sql:
            # Clean SQL: remove newlines, strip whitespace
            try:
                sql_clean = final_sql.replace('\n', ' ').strip()
            except Exception:
                sql_clean = str(final_sql).replace('\n', ' ').strip()
            sql_list.append(sql_clean)
        else:
            missing_count += 1
            # Use empty string as placeholder
            sql_list.append("")
    
    if missing_count > 0:
        print(f"Warning: {missing_count} entries missing final_sql field")
    
    # Write to output file (one SQL per line)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sql in sql_list:
            f.write(sql + '\n')
    
    print(f"Extracted {len(sql_list)} SQL statements to {output_file}")
    print(f"  - Valid SQL: {len(sql_list) - missing_count}")
    print(f"  - Missing SQL: {missing_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract final_sql from firstloop log JSON file")
    parser.add_argument("--log_file", type=str, required=True, help="Path to firstloop log JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output text file (one SQL per line)")
    
    args = parser.parse_args()
    extract_sql_from_log(args.log_file, args.output_file)

