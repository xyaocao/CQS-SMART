import argparse
import sys
from pathlib import Path
from pipeline_utils import PipelineConfig
from pipeline_runner import SinglerunPipeline, BatchPipeline
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.run_planner import get_table_paths  
from baseline.evaluation import get_spider_paths  

def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run the BaseMAD pipeline.")
    parser.add_argument("question", nargs="?", help="Natural language question (required for single mode)")
    parser.add_argument("db_id", nargs="?", help="Database id (required for single mode)")
    parser.add_argument("--input_mode", choices=["single", "batch"], help="Run a single question or a batch loop. Choices: single, batch.")
    parser.add_argument("--loop_mode", type=str, choices=["firstloop", "BaseMAD"], help="Choose a loop mode. Choices: firstloop, BaseMAD.")
    parser.add_argument("--dataset", default="spider", choices=["spider", "bird"], help="Choose the dataset to use. Choices: spider, bird.")
    parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Dataset split to use. Choices: dev, test.")
    parser.add_argument("--start", type=int, default=0, help="Start index within the examples (batch mode).")
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum number of examples to process (<=0 for all).")
    parser.add_argument("--planner_temperature", type=float, default=0.0, help="Planner LLM temperature.")
    parser.add_argument("--skeptic_temperature", type=float, default=0.0, help="Skeptic LLM temperature.")
    parser.add_argument("--reasoner_temperature", type=float, default=0.0, help="Reasoner LLM temperature.")
    parser.add_argument("--max_tokens", type=int, default=6000, help="Max tokens for each LLM call.")
    parser.add_argument("--examples_path", type=str, help="Override path to dataset examples JSON (batch mode).")
    parser.add_argument("--tables_path", type=str, help="Override path to tables metadata JSON.")
    parser.add_argument("--db_root", type=str, help="Override path to database root folder.")
    parser.add_argument("--log_path", type=str, help="Path to JSONL log output.")
    parser.add_argument( "--skeptic_questions_path", type=str, default="Exp/critical_questions.txt", help="Path to the skeptic critical-questions text file.")
    parser.add_argument("--max_debate_rounds", type=int, default=3, help="Maximal rounds of the debate.")
    parser.add_argument("--enable_early_termination", action="store_true", default=True, help="Enable early termination if views stabilize (BaseMAD optimization, default: True).")
    parser.add_argument("--disable_early_termination", action="store_false", dest="enable_early_termination", help="Disable early termination (BaseMAD).")
    parser.add_argument("--min_debate_rounds", type=int, default=2, help="Minimum debate rounds before checking for stability (BaseMAD optimization, default: 2).")
    parser.add_argument("--save_schema", action="store_true", help="Persist schema text for each example in the log.")

    args = parser.parse_args()

    if args.input_mode == "single" and (not args.question or not args.db_id):
        parser.error("question and db_id are required when --input_mode=single")

    examples, tables, db_root, _ = get_spider_paths(args.split)
    tables_path_str = args.tables_path or get_table_paths(args.dataset, args.split, args.tables_path)
    examples_path = Path(args.examples_path) if args.examples_path else Path(examples)
    tables_path = Path(tables_path_str)
    db_root = Path(args.db_root) if args.db_root else Path(db_root)

    if args.log_path:
        log_path = Path(args.log_path)
    else:
        default_log = ("Exp/logs/BaseMAD/singlerun_log.json"
                       if args.input_mode == "single"
                       else "Exp/logs/BaseMAD/batchrun_log.json"
                       )
        log_path = Path(default_log)
    
    return PipelineConfig(
        command_line = "".join(sys.argv),
        dataset = args.dataset,
        split = args.split,
        start = args.start,
        max_examples=args.max_examples,
        planner_temperature=args.planner_temperature,
        max_tokens=args.max_tokens,
        skeptic_temperature=args.skeptic_temperature,
        reasoner_temperature=args.reasoner_temperature,
        log_path=log_path,
        save_schema=args.save_schema,
        skeptic_questions_path=Path(args.skeptic_questions_path),
        max_debate_rounds = args.max_debate_rounds,
        enable_early_termination=args.enable_early_termination,
        min_debate_rounds=args.min_debate_rounds,
        input_mode=args.input_mode,
        loop_mode="BaseMAD",
        question=args.question,
        db_id=args.db_id,
        examples_path=examples_path,
        tables_path=tables_path,
        db_root=db_root,
    )

def main():
    config = parse_args()
    if config.input_mode == "single":
        runner = SinglerunPipeline(config)
        runner.run()
    else:
        batch_runner = BatchPipeline(config)
        batch_runner.run()

if __name__ == "__main__":
    main()



