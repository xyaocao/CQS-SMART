"""
Few-shot example retriever for online schema linking.

Uses sentence_transformers (same as RSL-SQL offline) to find similar QA pairs
based on Euclidean distance between question embeddings.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class FewShotExampleRetriever:
    """
    Retrieves few-shot examples using sentence_transformers similarity.

    Matches the offline RSL-SQL pipeline by using the same model and QA pairs
    to ensure consistent example selection between online and offline modes.
    """

    def __init__(self, dataset: str = "bird", k_shot: int = 3):
        """
        Initialize the few-shot example retriever.

        Args:
            dataset: "bird" or "spider" - determines which QA pairs and model to use
            k_shot: Default number of examples to retrieve (can be overridden per call)
        """
        self.dataset = dataset.lower()
        self.k_shot = k_shot

        # Use local schema_linking folder (relative to this file)
        schema_linking_dir = Path(__file__).resolve().parent / "schema_linking"

        # Set paths based on dataset
        if self.dataset == "bird":
            self.qa_path = schema_linking_dir / "bird" / "few_shot" / "QA.json"
            self.model_path = schema_linking_dir / "bird" / "few_shot" / "sentence_transformers"
        else:
            self.qa_path = schema_linking_dir / "spider" / "few_shot" / "QA.json"
            self.model_path = schema_linking_dir / "spider" / "few_shot" / "sentence_transformers"

        # Embedding cache path (store next to QA.json)
        self.cache_path = str(self.qa_path).replace('.json', '_embeddings.npy')

        # Load model and data
        self.bert_model = None
        self.train_data = None
        self.train_questions = None
        self.train_embeddings = None

        self._load_resources()

    def _load_resources(self):
        """Load the sentence transformer model and QA data."""
        # Lazy import to avoid loading if not used
        from sentence_transformers import SentenceTransformer

        # Load QA pairs
        if not self.qa_path.exists():
            raise FileNotFoundError(f"QA.json not found at {self.qa_path}")

        with open(self.qa_path, 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)

        self.train_questions = [item['question'] for item in self.train_data]

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Sentence transformer model not found at {self.model_path}")

        print(f"Loading sentence transformer model from {self.model_path}...")
        self.bert_model = SentenceTransformer(str(self.model_path), device="cpu")

        # Load or compute embeddings
        self.train_embeddings = self._get_or_compute_embeddings()
        print(f"Loaded {len(self.train_questions)} QA pairs for few-shot retrieval ({self.dataset})")

    def _get_or_compute_embeddings(self) -> np.ndarray:
        """
        Get embeddings from cache or compute and cache them.

        Returns:
            numpy array of shape (num_questions, embedding_dim)
        """
        if os.path.exists(self.cache_path):
            print(f"Loading cached embeddings from {self.cache_path}")
            return np.load(self.cache_path)

        print(f"Computing embeddings for {len(self.train_questions)} questions (first run)...")
        embeddings = self.bert_model.encode(
            self.train_questions,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Cache for future runs
        np.save(self.cache_path, embeddings)
        print(f"Cached embeddings to {self.cache_path}")

        return embeddings

    def get_examples(self, question: str, k: Optional[int] = None) -> List[Dict]:
        """
        Find k most similar examples to the given question.

        Args:
            question: The question to find similar examples for
            k: Number of examples to return (uses self.k_shot if None)

        Returns:
            List of k most similar QA pairs as dicts with 'question' and 'sql' keys
        """
        k = k or self.k_shot

        # Encode the target question
        target_embedding = self.bert_model.encode(
            question,
            show_progress_bar=False,
            convert_to_numpy=True
        ).reshape(1, -1)

        # Compute Euclidean distances to all training questions
        distances = euclidean_distances(target_embedding, self.train_embeddings)[0]

        # Get indices of k nearest neighbors
        top_k_indices = np.argsort(distances)[:k]

        return [self.train_data[i] for i in top_k_indices]

    def format_examples(self, examples: List[Dict]) -> str:
        """
        Format examples into the standard prefix string.

        Matches the offline RSL-SQL format exactly:
        - Prefix header
        - Each example as "### question\nsql"

        Args:
            examples: List of QA pairs from get_examples()

        Returns:
            Formatted string ready to use as few-shot prompt prefix
        """
        prefix = "### Some example pairs of question and corresponding SQL query are provided based on similar problems:"

        for ex in examples:
            # Clean up question and SQL (remove newlines, strip whitespace)
            q = ex['question'].replace('\n', ' ').strip()
            sql = ex.get('sql', ex.get('query', '')).replace('\n', ' ').strip()
            prefix += f"\n\n### {q}\n{sql}"

        return prefix

    def get_formatted_examples(self, question: str, k: Optional[int] = None) -> str:
        """
        Convenience method to get formatted examples in one call.

        Args:
            question: The question to find similar examples for
            k: Number of examples to return (uses self.k_shot if None)

        Returns:
            Formatted string with k similar examples
        """
        examples = self.get_examples(question, k)
        return self.format_examples(examples)


# Convenience function for one-off usage
def get_few_shot_examples(
    question: str,
    dataset: str = "bird",
    k: int = 3
) -> str:
    """
    Get formatted few-shot examples for a question.

    Note: This creates a new retriever each call, which is slow.
    For batch processing, create a FewShotExampleRetriever instance
    and reuse it.

    Args:
        question: The question to find similar examples for
        dataset: "bird" or "spider"
        k: Number of examples to return

    Returns:
        Formatted string with k similar examples
    """
    retriever = FewShotExampleRetriever(dataset=dataset, k_shot=k)
    return retriever.get_formatted_examples(question)


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="Test few-shot example retrieval")
    parser.add_argument("question", help="Question to find examples for")
    parser.add_argument("--dataset", choices=["bird", "spider"], default="bird")
    parser.add_argument("--k", type=int, default=3, help="Number of examples")

    args = parser.parse_args()

    print(f"Finding {args.k} similar examples for: {args.question}")
    print(f"Dataset: {args.dataset}")
    print("-" * 50)

    retriever = FewShotExampleRetriever(dataset=args.dataset, k_shot=args.k)
    examples = retriever.get_examples(args.question)

    print(f"\nFound {len(examples)} examples:")
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Q: {ex['question'][:100]}...")
        print(f"SQL: {ex['sql'][:100]}...")

    print("\n" + "=" * 50)
    print("Formatted output:")
    print(retriever.format_examples(examples))
