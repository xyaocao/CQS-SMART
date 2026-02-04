"""
Enhanced integration utilities to use RSL-SQL ppl_dev.json data in firstloop system.

Key improvements:
1. Uses ppl_loader_new which includes few-shot examples by default
2. Adds RSL-style prompt building for direct SQL generation
3. Better schema linking validation
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add parent directory to path for imports
utils_dir = Path(__file__).parent
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

from ppl_loader_new import (
    load_ppl_data,
    convert_ppl_to_examples,
    build_enhanced_schema_text,
    build_schema_text_without_examples,
    build_rsl_style_prompt,
    get_ppl_example_by_index,
    get_ppl_example_by_db_and_question,
    extract_schema_linking_info,
    validate_plan_against_schema_linking,
)


class PPLDataLoaderNew:
    """
    Enhanced loader for RSL-SQL ppl_dev.json data.

    Key changes from original:
    - get_enhanced_schema() now includes few-shot examples by default
    - Added get_rsl_style_prompt() for direct SQL generation
    - Added validation utilities for schema linking compliance
    """

    def __init__(self, ppl_json_path: str):
        """Initialize with path to ppl_dev.json."""
        self.ppl_json_path = ppl_json_path
        self.ppl_data: Optional[List[Dict[str, Any]]] = None
        self._load()

    def _load(self):
        """Load ppl_dev.json data."""
        self.ppl_data = load_ppl_data(self.ppl_json_path)
        # Convert 'db' to 'db_id' for consistency
        self.ppl_data = convert_ppl_to_examples(self.ppl_data)

    def get_examples(self) -> List[Dict[str, Any]]:
        """Get examples in standard format (with db_id instead of db)."""
        return self.ppl_data or []

    def get_enhanced_schema(self, index: int, include_examples: bool = True) -> Optional[str]:
        """
        Get enhanced schema text for a specific example by index.

        IMPORTANT: include_examples=True by default (changed from original).
        This ensures few-shot examples are always available for SQL generation.

        Args:
            index: Example index
            include_examples: Whether to include few-shot examples (default True)

        Returns:
            Enhanced schema text string, or None if index invalid
        """
        if not self.ppl_data or index < 0 or index >= len(self.ppl_data):
            return None

        item = self.ppl_data[index]
        return build_enhanced_schema_text(item, include_examples=include_examples)

    def get_enhanced_schema_by_db_question(
        self,
        db_id: str,
        question: str,
        include_examples: bool = True
    ) -> Optional[str]:
        """
        Get enhanced schema text by matching db_id and question.

        Args:
            db_id: Database ID
            question: Natural language question
            include_examples: Whether to include few-shot examples (default True)

        Returns:
            Enhanced schema text string, or None if not found
        """
        if not self.ppl_data:
            return None

        item = get_ppl_example_by_db_and_question(self.ppl_data, db_id, question)
        if not item:
            return None

        return build_enhanced_schema_text(item, include_examples=include_examples)

    def get_rsl_style_prompt(self, index: int, question: str = None) -> Optional[str]:
        """
        Get RSL-SQL style prompt for direct SQL generation.

        This creates a prompt identical to RSL-SQL's step_1_preliminary_sql.py
        which achieves ~82% accuracy on Spider.

        Args:
            index: Example index
            question: Optional question override (uses example's question if not provided)

        Returns:
            RSL-style prompt string, or None if index invalid
        """
        if not self.ppl_data or index < 0 or index >= len(self.ppl_data):
            return None

        item = self.ppl_data[index]
        q = question or item.get("question", "")
        return build_rsl_style_prompt(item, q)

    def get_example(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific example by index."""
        if not self.ppl_data or index < 0 or index >= len(self.ppl_data):
            return None
        return self.ppl_data[index]

    def find_example(self, db_id: str, question: str) -> Optional[Dict[str, Any]]:
        """Find example by db_id and question."""
        if not self.ppl_data:
            return None
        return get_ppl_example_by_db_and_question(self.ppl_data, db_id, question)

    def get_schema_linking_info(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get schema linking information for an example.

        Returns dict with:
        - tables: List of relevant tables
        - columns: List of relevant columns
        - has_evidence: Whether evidence/definition is available
        - has_examples: Whether few-shot examples are available
        """
        if not self.ppl_data or index < 0 or index >= len(self.ppl_data):
            return None

        item = self.ppl_data[index]
        return extract_schema_linking_info(item)

    def validate_plan(self, index: int, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a planner's output against schema linking results.

        Args:
            index: Example index
            plan: Planner output to validate

        Returns:
            Dict with validation results:
            - valid: Boolean indicating if plan follows schema linking
            - extra_tables: Tables in plan not in schema linking
            - missing_tables: Tables in schema linking not in plan
            - warnings: List of warning messages
        """
        if not self.ppl_data or index < 0 or index >= len(self.ppl_data):
            return {"valid": True, "warnings": ["Index out of range"]}

        item = self.ppl_data[index]
        return validate_plan_against_schema_linking(plan, item)


def create_ppl_examples_file(ppl_json_path: str, output_path: str):
    """
    Convert ppl_dev.json to standard examples format (with db_id).
    This allows using ppl_dev.json directly as input to the firstloop system.
    """
    ppl_data = load_ppl_data(ppl_json_path)
    examples = convert_ppl_to_examples(ppl_data)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(examples)} examples from {ppl_json_path} to {output_path}")
    return output_path


# Backwards compatibility
PPLDataLoader = PPLDataLoaderNew
