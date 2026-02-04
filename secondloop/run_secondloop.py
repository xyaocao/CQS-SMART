"""
Simplified SecondLoop Runner.

Pipeline:
1. Planner → Plan
2. SQLGen → initial_sql (with optional --voting)
3. SQL Reviewer → verdict, confidence, issues
4. If needs_revision → SQLGenWithFeedback → revised_sql
5. (Optional --refine) Execute → if error → Refiner

LLM Calls:
- Basic: 3 calls (Planner + SQLGen + Reviewer)
- With revision: 4 calls
- With voting (N=3): 5-6 calls

Usage:
    # Basic (Spider)
    python secondloop/run_secondloop.py --dataset spider --split dev --start 0 --max_examples 50
    
    # Basic (BIRD)
    python secondloop/run_secondloop.py --dataset bird --split dev --start 0 --max_examples 50

    # With voting (Spider)
    python secondloop/run_secondloop.py --dataset spider --split dev --voting --voting_samples 3
    
    # With voting (BIRD)
    python secondloop/run_secondloop.py --dataset bird --split dev --voting --voting_samples 3

    # With execution refiner (Spider)
    python secondloop/run_secondloop.py --dataset spider --split dev --refine
    
    # With execution refiner (BIRD)
    python secondloop/run_secondloop.py --dataset bird --split dev --refine

    # Full features (Spider)
    python secondloop/run_secondloop.py --dataset spider --split dev --voting --refine --verify
    
    # Full features (BIRD)
    python secondloop/run_secondloop.py --dataset bird --split dev --voting --refine --verify

    # With Ollama (Spider)
    python secondloop/run_secondloop.py --use_ollama --ollama_model ticlazau/qwen2.5-coder-7b-instruct --dataset spider --split dev
    
    # With Ollama (BIRD)
    python secondloop/run_secondloop.py --use_ollama --ollama_model ticlazau/qwen2.5-coder-7b-instruct --dataset bird --split dev
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add paths
baseline_dir = Path(__file__).resolve().parent.parent  # Root folder
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
# Add Exp folder to path for accessing Exp modules if needed
exp_dir = baseline_dir / "Exp"
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))


from baseline.llm import LLMConfig, get_ollama_config
from baseline.exec_match import exec_sql, exec_match, resolve_db_path
from baseline.dataloader import get_schema_from_spider, get_schema_from_bird

from agents_secondloop import (
    PlannerAgentV2,
    SQLGenAgent,
    Skeptic,
    SQLGenWithFeedbackAgent,
    SQLRefinerAgent,
    SelfConsistencyVoter,
    SelfVerificationAgent,
)

from loop_engines_secondloop import (
    SecondLoopEngine,
    SecondLoopResult,
    DirectSQLEngine,
    HybridEngine,
)

from utils.ppl_integration_new import PPLDataLoaderNew


def load_examples(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_json_log(log_path: Path, record: Dict[str, Any]):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing = [existing]
        except:
            pass

    existing.append(record)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)


def safe_exec(db_path: str, sql: str):
    """Execute SQL and return (result, error)."""
    try:
        result = exec_sql(db_path, sql)
        return result, None
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Run Simplified SecondLoop")

    # Input
    parser.add_argument("question", nargs="?", help="Question for single mode")
    parser.add_argument("db_id", nargs="?", help="Database ID for single mode")
    parser.add_argument("--input_mode", choices=["single", "batch"], default="batch")

    # Dataset
    parser.add_argument("--dataset", choices=["spider", "bird"], default="spider")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=50)

    # Schema
    parser.add_argument("--use_ppl_schema", action="store_true", default=True)
    parser.add_argument("--ppl_json_path", type=str, default=None,
                        help="Path to PPL JSON file (can specify ppl_dev.json or ppl_dev_enhanced.json)")
    parser.add_argument("--no_enhanced_ppl", action="store_true", 
                        help="Force use of regular ppl_dev.json instead of enhanced version")

    # LLM
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--use_ollama", action="store_true")
    parser.add_argument("--ollama_model", type=str, default="ticlazau/qwen2.5-coder-7b-instruct")

    # Features
    parser.add_argument("--voting", action="store_true", help="Enable self-consistency voting")
    parser.add_argument("--voting_samples", type=int, default=3, help="Number of voting samples")
    parser.add_argument("--refine", action="store_true", help="Enable execution error refinement")
    parser.add_argument("--verify", action="store_true", help="Enable self-verification after review")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid routing")
    parser.add_argument("--complexity_threshold", type=int, default=4)

    # Logging
    parser.add_argument("--log_path", type=str, default=None)

    args = parser.parse_args()

    # === PATHS ===
    # Make paths relative to project root (baseline_dir)
    if args.dataset == "spider":
        db_root = baseline_dir / "Data" / "spider_data" / "database"
        enhanced_ppl = baseline_dir / "Schema-Linking" / "RSL-SQL-Spider" / "src" / "information" / "ppl_dev_enhanced.json"
        regular_ppl = baseline_dir / "Schema-Linking" / "RSL-SQL-Spider" / "src" / "information" / "ppl_dev.json"
        gold_path = baseline_dir / "Data" / "spider_data" / f"{args.split}.json"
    else:
        # BIRD structure: Data/BIRD/{split}/{split}_databases/{db_id}/{db_id}.sqlite
        db_root = baseline_dir / "Data" / "BIRD" / args.split / f"{args.split}_databases"
        enhanced_ppl = baseline_dir / "Schema-Linking" / "RSL-SQL-Bird" / "src" / "information" / "ppl_dev_enhanced.json"
        regular_ppl = baseline_dir / "Schema-Linking" / "RSL-SQL-Bird" / "src" / "information" / "ppl_dev.json"
        gold_path = baseline_dir / "Data" / "BIRD" / args.split / f"{args.split}.json"

    # Determine which PPL file to use
    if args.ppl_json_path:
        # User explicitly specified a path
        ppl_path = Path(args.ppl_json_path)
        # If relative path, make it relative to project root
        if not ppl_path.is_absolute():
            ppl_path = baseline_dir / ppl_path
        print(f"Using specified PPL: {ppl_path}")
    elif args.no_enhanced_ppl:
        # Force use of regular version
        ppl_path = regular_ppl
        print(f"Using regular PPL (--no_enhanced_ppl): {ppl_path}")
    else:
        # Default: auto-detect (prefer enhanced if available)
        if enhanced_ppl.exists():
            ppl_path = enhanced_ppl
            print(f"Auto-detected enhanced PPL: {ppl_path}")
        else:
            ppl_path = regular_ppl
            print(f"Auto-detected regular PPL: {ppl_path}")

    # Log path
    if args.log_path:
        log_path = Path(args.log_path)
        # If relative path, make it relative to project root
        if not log_path.is_absolute():
            log_path = baseline_dir / log_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        features = []
        if args.voting:
            features.append("voting")
        if args.refine:
            features.append("refine")
        if args.verify:
            features.append("verify")
        if args.hybrid:
            features.append("hybrid")
        feat_str = "_".join(features) if features else "base"
        log_path = baseline_dir / "secondloop" / "logs" / args.dataset / f"run_{feat_str}_{ts}.json"

    # === LOAD DATA ===
    ppl_loader = None
    if args.use_ppl_schema and ppl_path.exists():
        print(f"Loading PPL: {ppl_path}")
        ppl_loader = PPLDataLoaderNew(str(ppl_path))

    gold_data = None
    if gold_path.exists():
        print(f"Loading gold: {gold_path}")
        gold_data = load_examples(gold_path)

    # === LLM CONFIG ===
    if args.use_ollama:
        llm_config = get_ollama_config(
            model=args.ollama_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"Using Ollama: {args.ollama_model}")
    else:
        llm_config = LLMConfig(temperature=args.temperature, max_tokens=args.max_tokens)

    # === SETUP AGENTS ===
    print("Setting up agents...")
    planner = PlannerAgentV2(llm_config)
    sql_gen = SQLGenAgent(llm_config)
    sql_reviewer = Skeptic(llm_config)
    sql_gen_feedback = SQLGenWithFeedbackAgent(llm_config)
    refiner = SQLRefinerAgent(llm_config) if args.refine else None
    voter = SelfConsistencyVoter(llm_config) if args.voting else None
    verifier = SelfVerificationAgent(llm_config) if args.verify else None

    # === SETUP ENGINE ===
    print(f"Engine: voting={args.voting}, refine={args.refine}, verify={args.verify}, hybrid={args.hybrid}")

    full_engine = SecondLoopEngine(
        planner=planner,
        sql_gen=sql_gen,
        sql_reviewer=sql_reviewer,
        sql_gen_with_feedback=sql_gen_feedback,
        refiner=refiner,
        voter=voter,
        verifier=verifier,
        enable_voting=args.voting,
        voting_samples=args.voting_samples,
        enable_refiner=args.refine,
        enable_verification=args.verify,
        db_executor=safe_exec if args.refine else None,
    )

    if args.hybrid:
        direct_engine = DirectSQLEngine(sql_gen)
        engine = HybridEngine(
            direct_engine=direct_engine,
            full_engine=full_engine,
            complexity_threshold=args.complexity_threshold,
        )
    else:
        engine = full_engine

    # === SINGLE MODE ===
    if args.input_mode == "single":
        if not args.question or not args.db_id:
            print("Error: question and db_id required")
            sys.exit(1)

        if ppl_loader:
            ex = ppl_loader.find_example(args.db_id, args.question)
            if ex:
                schema = ppl_loader.get_enhanced_schema(ppl_loader.get_examples().index(ex))
            else:
                schema = get_schema_from_spider(args.db_id) if args.dataset == "spider" else get_schema_from_bird(args.db_id)
        else:
            schema = get_schema_from_spider(args.db_id) if args.dataset == "spider" else get_schema_from_bird(args.db_id)

        # Resolve database path (tries multiple candidates)
        db_path = resolve_db_path(str(db_root), args.db_id, split=args.split, dataset=args.dataset)
        if not db_path:
            # Fallback to original path construction
            # Resolve database path (tries multiple candidates)
            db_path = resolve_db_path(str(db_root), args.db_id, split=args.split, dataset=args.dataset)
        if not db_path:
            # Fallback to original path construction
            db_path = str(db_root / args.db_id / f"{args.db_id}.sqlite")
        result = engine.run(args.question, schema, db_path=db_path)

        print(f"\n=== PLAN ===\n{json.dumps(result.plan, indent=2)}")
        print(f"\n=== INITIAL SQL ===\n{result.initial_sql}")
        print(f"\n=== REVIEW ===\n{json.dumps(result.review, indent=2)}")
        print(f"\n=== REVISED SQL ===\n{result.revised_sql}")
        print(f"\n=== FINAL SQL ===\n{result.final_sql}")
        print(f"\n=== LATENCY ===\n{result.latency}")
        print(f"\nRevision applied: {result.revision_applied}")
        print(f"Refiner applied: {result.refiner_applied}")
        return

    # === BATCH MODE ===
    print(f"\nBatch: {args.dataset}/{args.split} [{args.start}:{args.start + args.max_examples})")
    print(f"Log: {log_path}")
    print("-" * 50)

    examples = ppl_loader.get_examples() if ppl_loader else load_examples(gold_path)
    examples = examples[args.start:args.start + args.max_examples]

    correct = 0
    total = 0

    for i, ex in enumerate(examples):
        idx = args.start + i
        question = ex.get("question", "")
        db_id = ex.get("db_id", "")

        # Gold SQL - BIRD uses "SQL" (uppercase), Spider uses "query" or "sql" (lowercase)
        if gold_data and idx < len(gold_data):
            gold_sql = gold_data[idx].get("SQL", gold_data[idx].get("query", gold_data[idx].get("sql", "")))
        else:
            gold_sql = ex.get("SQL", ex.get("query", ex.get("sql", "")))

        # Schema
        if ppl_loader:
            schema = ppl_loader.get_enhanced_schema(idx)
        else:
            schema = get_schema_from_spider(db_id) if args.dataset == "spider" else get_schema_from_bird(db_id)

        # Run - resolve database path (tries multiple candidates)
        db_path = resolve_db_path(str(db_root), db_id, split=args.split, dataset=args.dataset)
        if not db_path:
            # Fallback to original path construction
            db_path = str(db_root / db_id / f"{db_id}.sqlite")
        try:
            result = engine.run(question, schema, db_path=db_path)
            final_sql = result.final_sql
            latency = result.latency
            plan = result.plan
            initial_sql = result.initial_sql
            review = result.review
            revised_sql = result.revised_sql
            revision_applied = result.revision_applied
            refiner_applied = result.refiner_applied
            verification_applied = result.verification_applied
            verification = result.verification
            voting_info = result.voting_info
            debug = result.debug
        except Exception as e:
            final_sql = ""
            initial_sql = ""
            revised_sql = ""
            latency = {"error": str(e)}
            plan = {}
            review = {}
            revision_applied = False
            refiner_applied = False
            verification_applied = False
            verification = None
            voting_info = None
            debug = {"error": str(e)}

        # Evaluate
        is_match = False
        try:
            # Check if database file exists
            db_path_obj = Path(db_path)
            if not db_path_obj.exists():
                latency["exec_error"] = f"Database file not found: {db_path}"
                latency["db_path_attempted"] = str(db_path)
            else:
                # Store SQL queries for debugging
                latency["gen_sql_executed"] = final_sql
                latency["gold_sql_executed"] = gold_sql
                
                gen_rows = exec_sql(db_path, final_sql)
                gold_rows = exec_sql(db_path, gold_sql)
                
                # Store execution results for debugging
                latency["gen_rows_count"] = len(gen_rows) if gen_rows else 0
                latency["gold_rows_count"] = len(gold_rows) if gold_rows else 0
                latency["gen_rows_sample"] = str(gen_rows[:3]) if gen_rows else "[]"
                latency["gold_rows_sample"] = str(gold_rows[:3]) if gold_rows else "[]"
                
                is_match, match_details = exec_match(gen_rows, gold_rows, order_matters=False, match_mode="hard")
                if not is_match:
                    # Store match details for debugging
                    latency["match_details"] = match_details
                    # Also store full results for detailed debugging (limit size)
                    latency["gen_rows_full"] = str(gen_rows) if len(str(gen_rows)) < 1000 else str(gen_rows)[:1000] + "..."
                    latency["gold_rows_full"] = str(gold_rows) if len(str(gold_rows)) < 1000 else str(gold_rows)[:1000] + "..."
        except Exception as e:
            is_match = False
            latency["exec_error"] = str(e)
            latency["db_path_used"] = db_path

        if is_match:
            correct += 1
        total += 1
        accuracy = float(correct) / float(total) if total > 0 else 0.0

        # Log
        record = {
            "timestamp": datetime.now().isoformat(),
            "command_line": " ".join(sys.argv),
            "example_index": idx,
            "question": question,
            "db_id": db_id,
            "features": {
                "voting": args.voting,
                "refine": args.refine,
                "verify": args.verify,
                "hybrid": args.hybrid,
            },
            "latency": latency,
            "plan": plan,
            "sql_tracking": {
                "initial_sql": initial_sql,
                "revised_sql": revised_sql,
                "final_sql": final_sql,
                "revision_applied": revision_applied,
                "refiner_applied": refiner_applied,
                "verification_applied": verification_applied,
            },
            "review": review,
            "verification": verification,
            "exec_match": is_match,
            "accuracy_so_far": accuracy,
            "voting_info": voting_info,
            "debug": debug,
        }
        update_json_log(log_path, record)

        # Progress
        status = "+" if is_match else "X"
        conf = float(review.get("confidence", 0)) if isinstance(review, dict) else 0.0
        verdict = review.get("verdict", "?") if isinstance(review, dict) else "?"
        flags = ""
        if revision_applied:
            flags += "[R]"
        if refiner_applied:
            flags += "[F]"
        if verification_applied:
            v_match = verification.get("match", True) if verification else True
            flags += "[V+" if v_match else "[V-"
            flags += "]"
        print(f"[{idx:4d}] {status} Acc:{accuracy:.1%} Conf:{conf:.2f} {verdict[:3]:4s} {flags:8s} | {question[:45]}...")

    # Summary
    print("\n" + "=" * 60)
    print(f"SecondLoop Results:")
    print(f"  Dataset: {args.dataset}/{args.split}")
    print(f"  Features: voting={args.voting}, refine={args.refine}, verify={args.verify}, hybrid={args.hybrid}")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/total:.2%}")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
