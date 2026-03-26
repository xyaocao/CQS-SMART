import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List

def load_entries(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed_lines = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    parsed_lines.append(entry)
            except json.JSONDecodeError:
                continue
        return parsed_lines

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        if "entries" in parsed and isinstance(parsed["entries"], list):
            return [item for item in parsed["entries"] if isinstance(item, dict)]
        return [parsed]

    return []

def get_exec_match(entry: dict) -> Any:
    """
    Some logs (baseline) nest exec_match inside the inputs dictionary.
    This helper normalizes access and returns None when the flag is missing.
    """
    if "exec_match" in entry:
        return entry["exec_match"]
    inputs = entry.get("inputs")
    if isinstance(inputs, dict):
        return inputs.get("exec_match")
    return None


def get_example_index(entry: dict) -> int:
    if "example_index" in entry:
        return int(entry["example_index"])
    inputs = entry.get("inputs")
    if isinstance(inputs, dict) and "example_index" in inputs:
        return int(inputs["example_index"])
    return -1


def collect_mismatches(entries: Iterable[dict]) -> List[int]:
    indices = []
    for entry in entries:
        match = get_exec_match(entry)
        if match is False:
            indices.append(get_example_index(entry))
    return indices


def write_output(indices: List[int], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(str(i) for i in indices)
    out_path.write_text(contents, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="List example_index values where exec_match is false.")
    parser.add_argument("--log_path", type=Path, help="Path to evaluator log JSON.")
    parser.add_argument("--output", type=Path, help="Where to write the example_index list. Defaults to <log>.mismatches.txt",)
    args = parser.parse_args()

    entries = load_entries(args.log_path)
    indices = collect_mismatches(entries)

    output_path = args.output or args.log_path.with_suffix(args.log_path.suffix + ".mismatches.txt")
    write_output(indices, output_path)

    print(f"Found {len(indices)} mismatches. Wrote indices to {output_path}")


if __name__ == "__main__":
    main()

