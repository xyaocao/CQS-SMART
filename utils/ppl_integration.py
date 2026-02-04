"""
Integration utilities to use RSL-SQL ppl_dev.json data in firstloop system.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
utils_dir = Path(__file__).parent
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

from ppl_loader import (
    load_ppl_data,
    convert_ppl_to_examples,
    build_enhanced_schema_text,
    build_schema_text_with_examples,
    get_ppl_example_by_index,
    get_ppl_example_by_db_and_question,
)


class PPLDataLoader:
    """Loader for RSL-SQL ppl_dev.json data."""
    
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
    
    def get_enhanced_schema(self, index: int, include_examples: bool = False) -> Optional[str]:
        """Get enhanced schema text for a specific example by index."""
        if not self.ppl_data or index < 0 or index >= len(self.ppl_data):
            return None
        
        item = self.ppl_data[index]
        if include_examples:
            return build_schema_text_with_examples(item)
        else:
            return build_enhanced_schema_text(item)
    
    def get_enhanced_schema_by_db_question(
        self, 
        db_id: str, 
        question: str, 
        include_examples: bool = False
    ) -> Optional[str]:
        """Get enhanced schema text by matching db_id and question."""
        if not self.ppl_data:
            return None
        
        item = get_ppl_example_by_db_and_question(self.ppl_data, db_id, question)
        if not item:
            return None
        
        if include_examples:
            return build_schema_text_with_examples(item)
        else:
            return build_enhanced_schema_text(item)
    
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

