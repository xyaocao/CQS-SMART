"""Test script to verify BIRD schema format with evidence."""
import json
import sys
from pathlib import Path

# Add parent directories to path
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))

from utils.ppl_loader import build_enhanced_schema_text

def main():
    # Load BIRD data
    bird_file = Path("Data/BIRD/dev/ppl_dev_converted.json")
    with open(bird_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Test first example with evidence
    example = data[0]
    print("=" * 80)
    print("BIRD Example 1:")
    print(f"Question: {example['question']}")
    print(f"Has evidence: {bool(example.get('evidence'))}")
    print(f"Evidence: {example.get('evidence', 'N/A')}")
    print("\n" + "=" * 80)
    print("Generated Schema Text:")
    print("=" * 80)
    
    schema = build_enhanced_schema_text(example)
    lines = schema.split('\n')
    
    # Find Definition section
    def_lines = [i for i, l in enumerate(lines) if 'Definition' in l or 'definition' in l]
    if def_lines:
        print(f"\nDefinition section found at line {def_lines[0] + 1}")
        start = max(0, def_lines[0] - 2)
        end = min(len(lines), def_lines[0] + 3)
        print("\nContext around Definition:")
        for i in range(start, end):
            marker = ">>> " if i == def_lines[0] else "    "
            print(f"{marker}{i+1:4d}: {lines[i]}")
    
    print("\n" + "=" * 80)
    print("Full Schema (first 2000 chars):")
    print("=" * 80)
    print(schema[:2000])
    if len(schema) > 2000:
        print(f"\n... (truncated, total length: {len(schema)} chars)")

if __name__ == "__main__":
    main()

