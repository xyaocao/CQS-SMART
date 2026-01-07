import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_history_hints(path: Path) -> List[Dict[str, Any]]:
    """
    Load hint entries from a jsonl file.
    Each line: {"db_id": "...", "pattern": "...", "hint": {...}, "regex": false}
    """
    hints: List[Dict[str, Any]] = []
    if not path or not path.exists():
        return hints
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            db_id = str(obj.get("db_id", "")).lower().strip()
            pattern = str(obj.get("pattern", "")).strip()
            if not db_id or not pattern:
                continue
            hints.append(
                {
                    "db_id": db_id,
                    "pattern": pattern,
                    "regex": bool(obj.get("regex")),
                    "hint": obj.get("hint") or {},
                    "source": obj.get("source"),
                }
            )
        except Exception:
            continue
    return hints


def match_history_hint(hints: List[Dict[str, Any]], question: str, db_id: str) -> Optional[Dict[str, Any]]:
    """Return a single matching hint or None if ambiguous/no match."""
    if not hints:
        return None
    q = (question or "").lower()
    db = (db_id or "").lower()
    matches: List[Dict[str, Any]] = []
    for h in hints:
        if h.get("db_id") != db:
            continue
        pat = h.get("pattern", "")
        if not pat:
            continue
        try:
            if h.get("regex"):
                if re.search(pat, q, re.IGNORECASE):
                    matches.append(h)
            else:
                if pat.lower() in q:
                    matches.append(h)
        except re.error:
            continue
    if len(matches) == 1:
        return matches[0]
    return None

