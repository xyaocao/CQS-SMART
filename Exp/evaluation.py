import argparse
import sys
from pathlib import Path
from pipeline_utils import PipelineConfig  
from evaluator import MultiAgentEvaluator  
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline.llm import LLMConfig, get_hetero_llm_pool, list_hetero_llm_pools  


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Evaluate the multi-agent pipeline on Spider.")
    parser.add_argument("--dataset", default="spider", choices=["spider", "BIRD"], help="Dataset to evaluate.")
    parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Spider split.")
    parser.add_argument("--start", type=int, default=0, help="Start index within the examples.")
    parser.add_argument("--max_examples", type=int, default=2000, help="Maximum examples to evaluate (<=0 for all).")
    parser.add_argument("--planner_temperature", type=float, default=0.0, help="Planner LLM temperature.")
    parser.add_argument("--skeptic_temperature", type=float, default=0.0, help="Skeptic LLM temperature.")
    parser.add_argument("--reasoner_temperature", type=float, default=0.0, help="Reasoner LLM temperature.")
    parser.add_argument("--max_tokens", type=int, default=1200, help="Max tokens per LLM call.")
    parser.add_argument("--examples_path", type=str, help="Override path to dataset examples JSON.")
    parser.add_argument("--tables_path", type=str, help="Override path to tables metadata JSON.")
    parser.add_argument("--db_root", type=str, help="Override path to database root folder.")
    parser.add_argument("--gold_sql_path", type=str, help="Override path to split_gold.sql.")
    parser.add_argument("--log_path", type=str, default="Exp/logs/eval_log.json", help="Path to JSONL log output.")
    parser.add_argument("--skeptic_questions_path", type=str, default="Exp/critical_questions.txt", help="Path to the skeptic critical-questions text file.",)
    parser.add_argument("--save_schema", action="store_true", help="Persist schema text in the log.")
    parser.add_argument("--loop_mode", choices=["first", "BaseMAD", "heteroMAD", "ImprovedMAD"],default="first",
        help=(
            "Which loop engine to evaluate: "
            "'first' (planner->skeptic->reasoner), "
            "'BaseMAD' (planner->debate between defender and skeptic->reasoner), "
            "'heteroMAD' (BaseMAD with random LLM selection from pool), "
            "'ImprovedMAD' (BaseMAD-style debate + contract reasoner + validator + 2-candidate chooser)"
            ))
    parser.add_argument("--max_debate_rounds", type=int,default=3,help="Maximum number of Defender–Skeptic debate rounds for BaseMAD evaluation.",)
    parser.add_argument("--enable_early_termination",action="store_true",default=True,help="Enable early termination if views stabilize (BaseMAD optimization, default: True).",)
    parser.add_argument( "--disable_early_termination", action="store_false",dest="enable_early_termination", help="Disable early termination (BaseMAD).",)
    parser.add_argument( "--min_debate_rounds", type=int, default=2,help="Minimum debate rounds before checking for stability (BaseMAD optimization, default: 2).", )
    # For heteroMAD: specify LLM pool either by name or as comma-separated model names
    available_pools = ", ".join(list_hetero_llm_pools())
    parser.add_argument("--hetero_llm_pool", type=str, help=f"For heteroMAD: predefined pool name. Available: {available_pools}. Alternative: use --hetero_llm_models for custom models")
    parser.add_argument("--hetero_llm_models", type=str, help="For heteroMAD: comma-separated list of model names (e.g., 'Qwen/Qwen3-Next-80B-A3B-Instruct,Qwen/Qwen2.5-Coder-14B-Instruct'). Alternative to --hetero_llm_pool")
    args = parser.parse_args()

    # Build LLM pool for heteroMAD if specified
    hetero_llm_pool = None
    if args.loop_mode == "heteroMAD":
        
        if args.hetero_llm_pool:
            # Use predefined pool
            try:
                model_names = get_hetero_llm_pool(args.hetero_llm_pool)
            except ValueError as e:
                available = ", ".join(list_hetero_llm_pools())
                raise ValueError(f"{e}. Available pools: {available}")
        elif args.hetero_llm_models:
            # Use custom model names
            model_names = [m.strip() for m in args.hetero_llm_models.split(",")]
        else:
            available = ", ".join(list_hetero_llm_pools())
            raise ValueError(
                f"heteroMAD requires either --hetero_llm_pool (predefined pool) or --hetero_llm_models (custom models). "
                f"Available predefined pools: {available}"
            )
        
        if len(model_names) < 2:
            raise ValueError("heteroMAD requires at least 2 models in the pool")
        
        # Create LLM configs for each model in the pool
        # Note: temperature and max_tokens are placeholders here - they will be
        # overridden per-agent (planner, skeptic, defender, reasoner) in pipeline_utils.py
        # based on the agent-specific settings from args
        hetero_llm_pool = [
            LLMConfig(
                model=model_name,
                temperature=0.0,  # Placeholder - will be overridden per-agent
                max_tokens=1000,  # Placeholder - will be overridden per-agent
            )
            for model_name in model_names
        ]

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
        loop_mode=args.loop_mode,
        max_debate_rounds=args.max_debate_rounds,
        enable_early_termination=args.enable_early_termination,
        min_debate_rounds=args.min_debate_rounds,
        hetero_llm_pool=hetero_llm_pool,
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
