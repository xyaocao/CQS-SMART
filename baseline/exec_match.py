import os
import json
import sqlite3
import threading
from typing import List, Tuple, Any, Iterable, Set
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import product
import random

def project_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))

def ignore_errors_decode(b: bytes) -> str:
    return b.decode(errors='ignore')

def exec_sql(db_path: str, sql: str, timeout: float = 30.0) -> List[Tuple[Any, ...]]:
    """
    Execute SQL query with timeout protection.
    
    Args:
        db_path: Path to SQLite database file
        sql: SQL query to execute
        timeout: Maximum time in seconds to wait for query execution (default: 30.0)
    
    Returns:
        List of result rows (tuples)
    
    Raises:
        TimeoutError: If query execution exceeds timeout
        Exception: Other SQL execution errors
    """
    result_container = {'rows': None, 'error': None}
    
    def execute_in_thread():
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.text_factory = ignore_errors_decode
            cur = conn.cursor()
            cur.execute(sql)
            result_container['rows'] = cur.fetchall()
        except Exception as e:
            result_container['error'] = e
        finally:
            if conn:
                conn.close()
    
    thread = threading.Thread(target=execute_in_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Query is still running, timeout occurred
        raise TimeoutError(f"SQL query execution exceeded timeout of {timeout}s. Query: {sql[:100]}...")
    
    if result_container['error'] is not None:
        raise result_container['error']
    
    if result_container['rows'] is None:
        raise RuntimeError("SQL execution returned no results and no error")
    
    return result_container['rows']

def normalize_cell(value: Any) -> Any:
    """
    Bring SQL cell values to a comparable form.
    Floats are rounded to reduce tiny precision diffs; bytes decode to str.
    """
    if isinstance(value, float):
        # Spider answers rarely need >6 decimals; rounding avoids noise
        return round(value, 6)
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="ignore")
    return value

def canonicalize_row(row: Iterable[Any]) -> Tuple[Any, ...]:
    """
    Sort row values so column ordering differences do not impact equality.
    """
    normalized = [normalize_cell(v) for v in row]
    try:
        return tuple(sorted(normalized))
    except TypeError:
        # As a last resort compare via str() to keep determinism
        return tuple(sorted(str(v) for v in normalized))

def canonicalize_rows(rows: Iterable[Tuple[Any, ...]]) -> Counter:
    """
    Convert a sequence of SQL rows to a multiset representation that
    ignores both row order and column order.
    """
    return Counter(canonicalize_row(row) for row in rows)

def read_examples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_gold_sql(path: str) -> List[Tuple[str, str]]:
    """
    Parse dev_gold.sql lines as-is, preserving duplicates:
      <SQL><TAB><db_id>
    Returns a list of (sql, db_id) tuples.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gold SQL file not found: {path}")

    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            if "\t" in line:
                sql, db_id = line.rsplit("\t", 1)
            else:
                # Fallback for lines without tabs: assume last token is db_id
                parts = line.split()
                if len(parts) < 2:
                    continue
                db_id = parts[-1]
                sql = line[: line.rfind(db_id)].rstrip()
            sql_clean = sql.strip().rstrip(";")
            db_id_clean = db_id.strip()
            if not sql_clean or not db_id_clean:
                continue
            pairs.append((sql_clean, db_id_clean))

    if not pairs:
        raise ValueError(
            f"No gold SQL rows parsed from {path}. "
            "Verify the file format uses tab separators (<SQL>\\t<db_id>)."
        )
    return pairs

def get_spider_paths(split: str) -> tuple[str, str, str, str]:
    root = project_root()
    examples = os.path.join(root, "Data", "spider_data", f"{split}.json")
    tables = os.path.join(
        root, "Data", "spider_data", "test_tables.json" if split == "test" else "tables.json"
    )
    db_root = os.path.join(
        root, "Data", "spider_data", "test_database" if split == "test" else "database"
    )
    gold = os.path.join(root, "Data", "spider_data", f"{split}_gold.sql")
    return examples, tables, db_root, gold

def get_bird_paths(split: str) -> tuple[str, str, str, str]:
    """Get default paths for BIRD dataset."""
    root = project_root()
    examples = os.path.join(root, "Data", "BIRD", split, f"{split}.json")
    tables = os.path.join(root, "Data", "BIRD", split, f"{split}_tables.json")
    db_root = os.path.join(root, "Data", "BIRD", split, f"{split}_databases")
    gold = os.path.join(root, "Data", "BIRD", split, f"{split}.sql")
    return examples, tables, db_root, gold

def get_dataset_paths(dataset: str, split: str) -> tuple[str, str, str, str]:
    """Get default paths for a given dataset and split."""
    dataset_lower = dataset.lower()
    if dataset_lower == "spider":
        return get_spider_paths(split)
    elif dataset_lower == "bird":
        return get_bird_paths(split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported: 'spider', 'bird'")

def resolve_db_path(db_root: str, db_id: str, split: str, dataset: str = "spider") -> str | None:
    """
    Try multiple common layouts to locate the SQLite file for a db_id.
    Priority:
      1) db_root/db_id/db_id.sqlite
      2) db_root/db_id.sqlite
      3) Standard dataset locations by split
    Returns the first existing path or None.
    """
    candidates: list[str] = []

    # As provided by args
    candidates.append(os.path.join(db_root, db_id, f"{db_id}.sqlite"))
    candidates.append(os.path.join(db_root, f"{db_id}.sqlite"))

    # Standard dataset locations by split
    root = project_root()
    dataset_lower = dataset.lower()
    
    if dataset_lower == "spider":
        split_pref = ["test_database", "database"] if split == "test" else ["database", "test_database"]
        for folder in split_pref:
            base = os.path.join(root, "Data", "spider_data", folder)
            candidates.append(os.path.join(base, db_id, f"{db_id}.sqlite"))
            candidates.append(os.path.join(base, f"{db_id}.sqlite"))
    elif dataset_lower == "bird":
        # BIRD structure: Data/BIRD/{split}/{split}_databases/{db_id}/{db_id}.sqlite
        base = os.path.join(root, "Data", "BIRD", split, f"{split}_databases")
        candidates.append(os.path.join(base, db_id, f"{db_id}.sqlite"))
        candidates.append(os.path.join(base, f"{db_id}.sqlite"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ============================================================================
# Soft and Partial Execution Matching Functions
# Integrated from Exp/exec_match.py
# ============================================================================

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    """Permute a tuple according to a permutation tuple."""
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    """Sort row values to ignore column ordering."""
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> Tuple[bool, str]:
    """
    Quick rejection check: compare unordered rows.
    Returns (is_match, details).
    """
    res = False
    details = ""
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        if s1 == s2:
            res = True
        else:
            details = (
                f"Results are not identical in ORDERED comparison\nGT:{s1}\nPT:{s2}"
            )
    else:
        if set(s1) == set(s2):
            res = True
        else:
            details = (
                f"Results are not identical in UNORDERED comparison\nGT:{s1}\nPT:{s2}"
            )
    return res, details


def multiset_eq(l1: List, l2: List) -> bool:
    """Check if two lists are equivalent multisets (same elements, same counts)."""
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    """
    Generate constrained column permutations for matching.
    Uses sampling to reduce the permutation space.
    """
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # Sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)
        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


def format_result(result: List[Tuple]) -> List[Tuple]:
    """Convert all Decimal values to float for comparison."""
    return [
        tuple(float(e) if isinstance(e, Decimal) else e for e in row) for row in result
    ]


def result_eq(
    result1: List[Tuple],
    result2: List[Tuple],
    order_matters: bool = True,
    is_hard: bool = True,
    is_partial: bool = False,
) -> Tuple[bool, str]:
    """
    Compare two SQL result sets with different matching modes.
    
    Args:
        result1: First result set (typically gold/expected)
        result2: Second result set (typically predicted)
        order_matters: Whether row order matters
        is_hard: If True, use hard matching (exact match with column permutation)
        is_partial: If True, use partial matching (result2 is subset of result1)
    
    Returns:
        Tuple of (is_match, details_string)
    
    Matching modes:
        - Hard (is_hard=True): Exact match allowing column/row permutations
        - Soft (is_hard=False, is_partial=False): result2 must contain all columns from result1
        - Partial (is_hard=False, is_partial=True): result2 is a subset of result1
    """
    result1 = format_result(result1)
    result2 = format_result(result2)
    details = "\n"
    
    if len(result1) == 0 and len(result2) == 0:
        return True, details

    # If length is not the same, they are definitely different bag of rows
    if len(result1) != len(result2):
        return False, details

    num_cols = len(result1[0])

    # For hard match: results must have the same number of columns
    if is_hard and len(result2[0]) != num_cols:
        details += f"Results do not have the same number of columns, GT: {num_cols}, PT: {len(result2[0])}"
        return False, details

    # Unorder each row and compare whether the denotation is the same
    # This can already find most pairs of denotations that are different
    if is_hard:
        quick_match, quick_details = quick_rej(result1, result2, order_matters)
        if not quick_match:
            return False, quick_details

        # The rest of the problem is in fact more complicated than one might think
        # We want to find a permutation of column order and a permutation of row order,
        # s.t. result_1 is the same as result_2
        # We return true if we can find such column & row permutations
        # and false if we cannot
        tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

        # On a high level, we enumerate all possible column permutations that might make result_1 == result_2
        # We decrease the size of the column permutation space by the function get_constraint_permutation
        # If one of the permutation make result_1, result_2 equivalent, then they are equivalent
        for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
            if len(perm) != len(set(perm)):
                continue
            if num_cols == 1:
                result2_perm = result2
            else:
                result2_perm = [permute_tuple(element, perm) for element in result2]
            if order_matters:
                if result1 == result2_perm:
                    details += f"ORDERED: FOUND matched permutation\nGT: {result1}\nPT: {result2}\nPT_perm: {result2_perm}"
                    return True, details
            else:
                # In fact the first condition must hold if the second condition holds
                # but the first is way more efficient implementation-wise
                # and we use it to quickly reject impossible candidates
                if set(result1) == set(result2_perm) and multiset_eq(
                    result1, result2_perm
                ):
                    details += f"UNORDERED: FOUND matched permutation\nGT: {set(result1)}\nPT: {set(result2)}\nPT_perm: {set(result2_perm)}"
                    return True, details

        if order_matters:
            details += f"Results are not identical in ORDERED comparison\nGT:{tab1_sets_by_columns}\nPT:{result2}"
        else:
            details += f"Results are not identical in UNORDERED comparison\nGT:{tab1_sets_by_columns}\nPT:{result2}"
        return False, details

    # For soft cases, we only check whether result_2 has all columns in result_1, even it might have more columns return
    # We can columnize the result_1 and result_2, and check whether result_2 has all columns in result_1
    hard_res = result_eq(result1, result2, order_matters, is_hard=True)
    _hard_res = hard_res[0]
    if _hard_res:
        return hard_res
    else:
        # Handle cases where column counts might differ (for soft/partial matching)
        if len(result1) == 0:
            return True, details  # Both empty (already checked above)
        if len(result2) == 0:
            return False, details  # result1 has rows but result2 is empty
        
        columnized_result_1 = list(zip(*result1))
        columnized_result_2 = list(zip(*result2))
        
        if not is_partial:
            # Soft match: result2 must contain all columns from result1
            if order_matters:
                for i, col in enumerate(columnized_result_1, start=1):
                    if len(col) > 0:
                        if col not in columnized_result_2:
                            details += f"Results are missing in SOFT comparison\nGT column {i}:{columnized_result_1[i-1]} is not in PT columns"
                            return False, details
                details += f"Results are identical in SOFT comparison\nGT:{columnized_result_1}\nPT:{columnized_result_2}"
                return True, details
            else:
                for i, col in enumerate(columnized_result_1, start=1):
                    if len(col) > 0:
                        if set(col) not in [set(c) for c in columnized_result_2]:
                            details += f"Results are missing in SOFT comparison\nGT column {i}:{columnized_result_1[i-1]} is not in PT columns"
                            return False, details
                details += f"Results are identical in SOFT comparison\nGT:{columnized_result_1}\nPT:{columnized_result_2}"
                return True, details
        
        # Partial match: result2 is a subset of result1
        # First try soft matching
        _soft_res = result_eq(result1, result2, order_matters, is_hard=False, is_partial=False)
        if _soft_res[0]:
            return _soft_res
        
        # Then check if result2 columns are all in result1 (partial match)
        for i, col in enumerate(columnized_result_2, start=1):
            if len(col) > 0:
                if set(col) not in [set(c) for c in columnized_result_1]:
                    details += f"Results are missing in PARTIAL comparison\nPT column {i}:{columnized_result_2[i-1]} is not in GT columns"
                    return False, details
        details += f"Results are partially identical in comparison\nGT:{columnized_result_1}\nPT:{columnized_result_2}"
        return True, details


def exec_match(
    gen_rows: List[Tuple[Any, ...]],
    gold_rows: List[Tuple[Any, ...]],
    order_matters: bool = False,
    match_mode: str = "hard",
) -> Tuple[bool, str]:
    """
    Convenience function to match execution results with different modes.
    
    Args:
        gen_rows: Generated/predicted SQL results
        gold_rows: Gold/expected SQL results
        order_matters: Whether row order matters (default: False for SQL)
        match_mode: One of "hard", "soft", "partial"
            - "hard": Exact match with column/row permutation (default)
            - "soft": Generated must contain all columns from gold
            - "partial": Generated is a subset of gold
    
    Returns:
        Tuple of (is_match, details_string)
    
    Example:
        >>> gen_rows = [(1, "Alice"), (2, "Bob")]
        >>> gold_rows = [("Alice", 1), ("Bob", 2)]
        >>> exec_match(gen_rows, gold_rows, match_mode="hard")
        (True, "...")
    """
    if match_mode == "hard":
        return result_eq(gold_rows, gen_rows, order_matters=order_matters, is_hard=True, is_partial=False)
    elif match_mode == "soft":
        return result_eq(gold_rows, gen_rows, order_matters=order_matters, is_hard=False, is_partial=False)
    elif match_mode == "partial":
        return result_eq(gold_rows, gen_rows, order_matters=order_matters, is_hard=False, is_partial=True)
    else:
        raise ValueError(f"Unknown match_mode: {match_mode}. Must be 'hard', 'soft', or 'partial'")