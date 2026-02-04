"""
Script to convert RSL-SQL ppl_dev.json to format compatible with firstloop system.
Usage:
    python utils/convert_ppl_for_firstloop.py \
        --input Schema-Linking/RSL-SQL-Spider/src/information/ppl_dev.json \
        --output Data/spider_data/ppl_dev_converted.json
"""
import argparse
from pathlib import Path
import sys

# Add parent directories to path
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))

from utils.ppl_integration import create_ppl_examples_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert RSL-SQL ppl_dev.json to firstloop-compatible format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input ppl_dev.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file (will have db_id instead of db)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {input_path} to {output_path}...")
    create_ppl_examples_file(str(input_path), str(output_path))
    print(f"✓ Conversion complete! {output_path}")


if __name__ == "__main__":
    main()

