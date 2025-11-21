import argparse
import sys
from pathlib import Path
from pipeline_utils import PipelineConfig  
from evaluator import MultiAgentEvaluator  


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Evaluate the multi-agent pipeline on Spider.")
    parser.add_argument("--dataset", default="spider", choices=["spider"], help="Dataset to evaluate.")
    parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Spider split.")
    parser.add_argument("--start", type=int, default=0, help="Start index within the examples.")
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum examples to evaluate (<=0 for all).")
    parser.add_argument("--planner_temperature", type=float, default=0.0, help="Planner LLM temperature.")
    parser.add_argument("--skeptic_temperature", type=float, default=0.2, help="Skeptic LLM temperature.")
    parser.add_argument("--reasoner_temperature", type=float, default=0.0, help="Reasoner LLM temperature.")
    parser.add_argument("--max_tokens", type=int, default=1200, help="Max tokens per LLM call.")
    parser.add_argument("--examples_path", type=str, help="Override path to dataset examples JSON.")
    parser.add_argument("--tables_path", type=str, help="Override path to tables metadata JSON.")
    parser.add_argument("--db_root", type=str, help="Override path to database root folder.")
    parser.add_argument("--gold_sql_path", type=str, help="Override path to split_gold.sql.")
    parser.add_argument("--log_path", type=str,default="Exp/logs/eval_log.json", help="Path to JSONL log output.",)
    parser.add_argument("--skeptic_questions_path", type=str, default="Exp/critical_questions.txt", help="Path to the skeptic critical-questions text file.",)
    parser.add_argument("--save_schema", action="store_true", help="Persist schema text in the log.")
    args = parser.parse_args()

    return PipelineConfig(
        dataset=args.dataset,
        split=args.split,
        start=args.start,
        max_examples=args.max_examples,
        planner_temperature=args.planner_temperature,
        max_tokens=args.max_tokens,
        skeptic_temperature=args.skeptic_temperature,
        reasoner_temperature=args.reasoner_temperature,
        log_path=Path(args.log_path),
        save_schema=args.save_schema,
        skeptic_questions_path=Path(args.skeptic_questions_path),
        command_line=" ".join(sys.argv),
        input_mode="batch",
        question=None,
        db_id=None,
        examples_path=Path(args.examples_path) if args.examples_path else None,
        tables_path=Path(args.tables_path) if args.tables_path else None,
        db_root=Path(args.db_root) if args.db_root else None,
        gold_sql_path=Path(args.gold_sql_path) if args.gold_sql_path else None,
    )


def main() -> None:
    config = parse_args()
    evaluator = MultiAgentEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
