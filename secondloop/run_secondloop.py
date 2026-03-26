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

    # === ONLINE SCHEMA LINKING ===
    # Process schema linking on-the-fly for each question (no pre-processed PPL files needed)

    # Single question with online schema linking (BIRD)
    python secondloop/run_secondloop.py --online_schema --dataset bird --input_mode single "What is the average salary?" "employee_db"

    # Batch with online schema linking (BIRD)
    python secondloop/run_secondloop.py --online_schema --dataset bird --split dev --start 0 --max_examples 10

    # With Ollama Server (port 11435) + online schema linking
    python secondloop/run_secondloop.py --online_schema --use_ollama_server --ollama_port 11435 --dataset bird --split dev

    # With custom Ollama server host/port
    python secondloop/run_secondloop.py --online_schema --use_ollama_server --ollama_host localhost --ollama_port 11435 --dataset bird
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


from baseline.llm import LLMConfig, get_ollama_config, get_ollama_server_config, get_exp_b_configs
from baseline.exec_match import exec_sql, exec_match, resolve_db_path

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
from utils.ppl_loader_new import build_enhanced_schema_text
from online_schema_linking import OnlineSchemaLinker, create_online_linker


def load_examples(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_spider_test_gold(path: Path) -> list:
    """Load Spider test_gold.sql (tab-separated SQL\tdb_id per line)."""
    gold = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                gold.append(parts[0])  # SQL query only
    return gold


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
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
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
    parser.add_argument("--use_ollama", action="store_true", help="Use local Ollama (port 11434)")
    parser.add_argument("--use_ollama_server", action="store_true", help="Use server Ollama (port 11435)")
    parser.add_argument("--ollama_model", type=str, default="ticlazau/qwen2.5-coder-7b-instruct")
    parser.add_argument("--ollama_host", type=str, default="localhost", help="Host for server Ollama")
    parser.add_argument("--ollama_port", type=int, default=11435, help="Port for server Ollama")

    # Online Schema Linking
    parser.add_argument("--online_schema", action="store_true",
                        help="Enable online schema linking (process each question on-the-fly)")

    # Features
    parser.add_argument("--voting", action="store_true", help="Enable self-consistency voting")
    parser.add_argument("--voting_samples", type=int, default=3, help="Number of voting samples")
    parser.add_argument("--refine", action="store_true", help="Enable execution error refinement")
    parser.add_argument("--verify", action="store_true", help="Enable self-verification after review")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid routing")
    parser.add_argument("--complexity_threshold", type=int, default=4)

    # Ablation flags
    parser.add_argument("--index_file", type=str, default=None,
                        help="JSON file with list of example indices to run")
    parser.add_argument("--skip_planner", action="store_true",
                        help="Ablation: skip planner agent")
    parser.add_argument("--skip_reviewer", action="store_true",
                        help="Ablation: skip reviewer and feedback")
    parser.add_argument("--no_few_shot", action="store_true",
                        help="Ablation: disable few-shot examples")
    parser.add_argument("--no_column_meaning", action="store_true",
                        help="Ablation: disable column meanings in schema")
    parser.add_argument("--no_schema_check", action="store_true",
                        help="Ablation: disable deterministic schema consistency check")

    # Experiment flags
    parser.add_argument("--exp_a", action="store_true",
                        help="Experiment A: watcher-aware prompts (reviewer-only config). "
                             "All agents told a separate AI model will evaluate their output. "
                             "Uses current LLM config. Automatically sets --skip_planner.")
    parser.add_argument("--exp_a_codefence", action="store_true",
                        help="Experiment A + code-fence: combines watcher-aware prompts with "
                             "explicit ```sql``` fence output instruction. Use this when running "
                             "DeepSeek-R1 as the backbone to prevent stuck chain-of-thought loops. "
                             "Automatically sets --skip_planner.")
    parser.add_argument("--exp_b", action="store_true",
                        help="Experiment B: heterogeneous LLMs (reviewer-only config). "
                             "SQLGen+Feedback=Qwen3-Next-80B, Reviewer=DeepSeek-R1-Distill-14B. "
                             "Automatically sets --skip_planner.")
    parser.add_argument("--exp_b_reversed", action="store_true",
                        help="Experiment B reversed: heterogeneous LLMs (reviewer-only config). "
                             "SQLGen+Feedback=DeepSeek-R1-Distill-14B, Reviewer=Qwen3-Next-80B. "
                             "Automatically sets --skip_planner.")
    parser.add_argument("--exp_b_reversed_codefence", action="store_true",
                        help="Experiment B reversed with code-fence output instruction. "
                             "Same as --exp_b_reversed but adds explicit ```sql``` fence instruction "
                             "to SQLGen and SQLGenWithFeedback prompts to fix DeepSeek-R1 "
                             "stuck-loop extraction failures. Automatically sets --skip_planner.")

    # Logging
    parser.add_argument("--log_path", type=str, default=None)

    args = parser.parse_args()

    # === PATHS ===
    # Use local schema_linking folder
    schema_linking_dir = Path(__file__).resolve().parent / "schema_linking"

    # Make paths relative to project root (baseline_dir)
    if args.dataset == "spider":
        if args.split == "test":
            db_root = baseline_dir / "Data" / "spider_data" / "test_database"
        else:
            db_root = baseline_dir / "Data" / "spider_data" / "database"
        enhanced_ppl = schema_linking_dir / "spider" / "information" / "ppl_dev_enhanced.json"
        regular_ppl = schema_linking_dir / "spider" / "information" / "ppl_dev.json"
        gold_path = baseline_dir / "Data" / "spider_data" / f"{args.split}.json"
    else:
        # BIRD structure: Data/BIRD/{split}/{split}_databases/{db_id}/{db_id}.sqlite
        db_root = baseline_dir / "Data" / "BIRD" / args.split / f"{args.split}_databases"
        enhanced_ppl = schema_linking_dir / "bird" / "information" / "ppl_dev_enhanced.json"
        regular_ppl = schema_linking_dir / "bird" / "information" / "ppl_dev.json"
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
        if args.skip_planner:
            features.append("noplanner")
        if args.skip_reviewer:
            features.append("noreviewer")
        if args.no_few_shot:
            features.append("nofewshot")
        if args.no_column_meaning:
            features.append("nocolmeaning")
        if args.no_schema_check:
            features.append("noschemacheck")
        if args.exp_a:
            features.append("expA")
        if args.exp_a_codefence:
            features.append("expA_cf")
        if args.exp_b:
            features.append("expB")
        if args.exp_b_reversed:
            features.append("expB_reversed")
        if args.exp_b_reversed_codefence:
            features.append("expB_reversed_cf")
        feat_str = "_".join(features) if features else "base"
        log_path = baseline_dir / "secondloop" / "logs" / args.dataset / f"run_{feat_str}_{ts}.json"

    # === LOAD DATA ===
    ppl_loader = None
    if args.use_ppl_schema and ppl_path.exists():
        print(f"Loading PPL: {ppl_path}")
        ppl_loader = PPLDataLoaderNew(str(ppl_path))

    # For Spider test split, gold SQL is in test_gold.sql (separate from test.json)
    test_gold_sql = None
    if args.dataset == "spider" and args.split == "test":
        test_gold_path = baseline_dir / "Data" / "spider_data" / "test_gold.sql"
        if test_gold_path.exists():
            print(f"Loading Spider test gold SQL: {test_gold_path}")
            test_gold_sql = load_spider_test_gold(test_gold_path)

    gold_data = None
    if gold_path.exists():
        print(f"Loading gold: {gold_path}")
        gold_data = load_examples(gold_path)

    # === LLM CONFIG ===
    if args.use_ollama_server:
        llm_config = get_ollama_server_config(
            model=args.ollama_model,
            host=args.ollama_host,
            port=args.ollama_port,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"Using Ollama Server: {args.ollama_model} @ {args.ollama_host}:{args.ollama_port}")
    elif args.use_ollama:
        llm_config = get_ollama_config(
            model=args.ollama_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"Using Ollama: {args.ollama_model}")
    else:
        llm_config = LLMConfig(temperature=args.temperature, max_tokens=args.max_tokens)

    # === ONLINE SCHEMA LINKER ===
    online_linker = None
    if args.online_schema:
        print(f"Online schema linking enabled for {args.dataset}")
        online_linker = create_online_linker(
            dataset=args.dataset,
            use_ollama=args.use_ollama,
            use_ollama_server=args.use_ollama_server,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            llm_config=llm_config,
            k_shot=0 if args.no_few_shot else 3,
            split=args.split,
        )

    # === EXPERIMENT FLAGS: all use reviewer-only config (no planner) ===
    if args.exp_a or args.exp_a_codefence or args.exp_b or args.exp_b_reversed or args.exp_b_reversed_codefence:
        args.skip_planner = True

    # === SETUP AGENTS ===
    print("Setting up agents...")

    if args.exp_b or args.exp_b_reversed or args.exp_b_reversed_codefence:
        if args.exp_b_reversed_codefence:
            pool_name = "reversed"
            use_codefence = True
            label = "Exp B reversed+codefence"
        elif args.exp_b_reversed:
            pool_name = "reversed"
            use_codefence = False
            label = "Exp B reversed"
        else:
            pool_name = "default"
            use_codefence = False
            label = "Exp B"

        gen_config, reviewer_config = get_exp_b_configs(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            pool_name=pool_name,
        )
        print(f"{label} | Generator : {gen_config.model}")
        print(f"{label} | Reviewer  : {reviewer_config.model}")
        if use_codefence:
            print(f"{label} | Code-fence instruction: enabled")

        # aggressive_json=True only when DeepSeek-R1 is the reviewer
        # (strips <think> tokens before JSON parsing)
        deepseek_as_reviewer = "deepseek" in reviewer_config.model.lower()

        planner          = PlannerAgentV2(llm_config)           # skipped anyway
        sql_gen          = SQLGenAgent(gen_config, exp_a=False, codefence_ins=use_codefence)
        sql_reviewer     = Skeptic(reviewer_config, exp_a=False,
                                   aggressive_json=deepseek_as_reviewer)
        sql_gen_feedback = SQLGenWithFeedbackAgent(gen_config, exp_a=False,
                                                   codefence_ins=use_codefence)
        refiner          = SQLRefinerAgent(gen_config) if args.refine else None
        voter            = SelfConsistencyVoter(gen_config) if args.voting else None
        verifier         = SelfVerificationAgent(gen_config) if args.verify else None
    else:
        # Default / Exp A / Exp A+codefence: use the configured llm_config for all agents
        use_exp_a = args.exp_a or args.exp_a_codefence
        use_codefence = args.exp_a_codefence
        if args.exp_a_codefence:
            print(f"Exp A+codefence | Watcher prompts + code-fence output | Model: {llm_config.model}")
        elif args.exp_a:
            print(f"Exp A | Watcher prompts enabled | Model: {llm_config.model}")
        planner          = PlannerAgentV2(llm_config)
        sql_gen          = SQLGenAgent(llm_config, exp_a=use_exp_a, codefence_ins=use_codefence)
        sql_reviewer     = Skeptic(llm_config, exp_a=use_exp_a)
        sql_gen_feedback = SQLGenWithFeedbackAgent(llm_config, exp_a=use_exp_a,
                                                   codefence_ins=use_codefence)
        refiner          = SQLRefinerAgent(llm_config) if args.refine else None
        voter            = SelfConsistencyVoter(llm_config) if args.voting else None
        verifier         = SelfVerificationAgent(llm_config) if args.verify else None

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
        enable_planner=not args.skip_planner,
        enable_reviewer=not args.skip_reviewer,
        no_schema_check=args.no_schema_check,
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

        # Get schema using online linking or fallback methods
        schema_result = None
        if online_linker:
            print("Processing with online schema linking...")
            schema_result = online_linker.process_question(
                question=args.question,
                db_id=args.db_id,
                evidence="",  # Can be passed via additional arg if needed
            )
            # Use build_enhanced_schema_text for consistent formatting with offline mode
            ppl_dict = schema_result.to_ppl_dict()
            schema = build_enhanced_schema_text(
                ppl_dict,
                include_examples=not args.no_few_shot,
                include_column_meaning=not args.no_column_meaning,
            )
            print(f"Schema linking found tables: {schema_result.tables}")
            print(f"Schema linking found columns: {schema_result.columns[:5]}..." if len(schema_result.columns) > 5 else f"Schema linking found columns: {schema_result.columns}")
            if schema_result.preliminary_sql:
                print(f"Preliminary SQL: {schema_result.preliminary_sql[:100]}..." if len(schema_result.preliminary_sql) > 100 else f"Preliminary SQL: {schema_result.preliminary_sql}")
        elif ppl_loader:
            ex = ppl_loader.find_example(args.db_id, args.question)
            if ex:
                schema = ppl_loader.get_enhanced_schema(ppl_loader.get_examples().index(ex))
            else:
                # Fallback: build schema from database
                from online_schema_linking import OnlineSchemaLinker
                temp_linker = OnlineSchemaLinker(dataset=args.dataset, llm_config=llm_config)
                schema = (
                    "### Database Schema:\n" + temp_linker.get_table_infos(args.db_id) +
                    "\n### Sample Data:\n" + temp_linker.get_sample_data(args.db_id) +
                    "\n### Foreign Keys:\n" + temp_linker.get_foreign_key_infos(args.db_id)
                )
        else:
            # Fallback: build schema from database
            from online_schema_linking import OnlineSchemaLinker
            temp_linker = OnlineSchemaLinker(dataset=args.dataset, llm_config=llm_config)
            schema = (
                "### Database Schema:\n" + temp_linker.get_table_infos(args.db_id) +
                "\n### Sample Data:\n" + temp_linker.get_sample_data(args.db_id) +
                "\n### Foreign Keys:\n" + temp_linker.get_foreign_key_infos(args.db_id)
            )

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

        if schema_result and schema_result.preliminary_sql:
            print(f"\n=== PRELIMINARY SQL (from schema linking) ===\n{schema_result.preliminary_sql}")
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
    if args.online_schema:
        print("Mode: Online Schema Linking (processing each question on-the-fly)")
    if args.index_file:
        print(f"Index file: {args.index_file}")
    print(f"Log: {log_path}")
    print("-" * 50)

    # Load all examples
    if args.online_schema:
        all_examples = load_examples(gold_path) if gold_path.exists() else []
    else:
        all_examples = ppl_loader.get_examples() if ppl_loader else load_examples(gold_path)

    # Select examples: index file OR range
    if args.index_file:
        index_path = Path(args.index_file)
        if not index_path.is_absolute():
            index_path = baseline_dir / index_path
        with open(index_path, "r") as f:
            subset_indices = json.load(f)
        examples_with_idx = [(idx, all_examples[idx]) for idx in subset_indices if idx < len(all_examples)]
        # Apply start/max_examples to the subset list for easy partial runs
        examples_with_idx = examples_with_idx[args.start:args.start + args.max_examples]
        print(f"Selected {len(examples_with_idx)} examples from index file")
    else:
        start = args.start
        end = min(start + args.max_examples, len(all_examples))
        examples_with_idx = [(idx, all_examples[idx]) for idx in range(start, end)]

    correct = 0
    total = 0

    for seq_i, (idx, ex) in enumerate(examples_with_idx):
        question = ex.get("question", "")
        db_id = ex.get("db_id", "") if not args.online_schema else ex.get("db_id", "")
        # BIRD uses "evidence" field for domain knowledge
        evidence = ex.get("evidence", "") if args.dataset == "bird" else ""

        # Gold SQL - Spider test uses test_gold.sql; BIRD uses "SQL"; Spider dev uses "query"
        if test_gold_sql and idx < len(test_gold_sql):
            gold_sql = test_gold_sql[idx]
        elif gold_data and idx < len(gold_data):
            gold_sql = gold_data[idx].get("SQL", gold_data[idx].get("query", gold_data[idx].get("sql", "")))
        else:
            gold_sql = ex.get("SQL", ex.get("query", ex.get("sql", "")))

        # Schema - use online linking if enabled
        schema_result = None  # Store for logging
        schema_linking_latency = {}
        if online_linker:
            try:
                schema_link_start = time.perf_counter()
                schema_result = online_linker.process_question(
                    question=question,
                    db_id=db_id,
                    evidence=evidence,
                )
                # Use build_enhanced_schema_text for consistent formatting with offline mode
                ppl_dict = schema_result.to_ppl_dict()
                schema = build_enhanced_schema_text(
                    ppl_dict,
                    include_examples=not args.no_few_shot,
                    include_column_meaning=not args.no_column_meaning,
                )
                schema_linking_latency["schema_linking_sec"] = time.perf_counter() - schema_link_start
                # Get step-level latencies if available
                if hasattr(schema_result, 'latency') and schema_result.latency:
                    schema_linking_latency.update(schema_result.latency)
            except Exception as e:
                print(f"  [WARN] Online schema linking failed for {db_id}: {e}")
                # Fallback: build basic schema from database structure
                try:
                    schema = (
                        "### Database Schema:\n" +
                        online_linker.get_table_infos(db_id) +
                        "\n### Sample Data:\n" +
                        online_linker.get_sample_data(db_id) +
                        "\n### Foreign Keys:\n" +
                        online_linker.get_foreign_key_infos(db_id)
                    )
                except Exception as e2:
                    schema = f"Error loading schema for {db_id}: {e2}"
                schema_linking_latency["schema_linking_error"] = str(e)
        elif ppl_loader:
            schema = ppl_loader.get_enhanced_schema(idx, include_examples=not args.no_few_shot)
            if args.no_column_meaning:
                item = ppl_loader.get_example(idx)
                schema = build_enhanced_schema_text(
                    item,
                    include_examples=not args.no_few_shot,
                    include_column_meaning=False,
                )
        else:
            # Fallback: build schema from database using a temporary linker
            temp_linker = create_online_linker(
                dataset=args.dataset,
                use_ollama=args.use_ollama,
                use_ollama_server=args.use_ollama_server,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
                ollama_port=args.ollama_port,
            )
            schema = (
                "### Database Schema:\n" + temp_linker.get_table_infos(db_id) +
                "\n### Sample Data:\n" + temp_linker.get_sample_data(db_id) +
                "\n### Foreign Keys:\n" + temp_linker.get_foreign_key_infos(db_id)
            )

        # Run - resolve database path (tries multiple candidates)
        db_path = resolve_db_path(str(db_root), db_id, split=args.split, dataset=args.dataset)
        if not db_path:
            # Fallback to original path construction
            db_path = str(db_root / db_id / f"{db_id}.sqlite")

        # Track total time for entire pipeline
        pipeline_start = time.perf_counter()
        try:
            result = engine.run(question, schema, db_path=db_path)
            final_sql = result.final_sql
            latency = result.latency.copy() if result.latency else {}
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

        # Merge schema linking latency with engine latency
        if schema_linking_latency:
            latency.update(schema_linking_latency)

        # Calculate total pipeline time (schema linking + engine)
        latency["pipeline_total_sec"] = time.perf_counter() - pipeline_start + schema_linking_latency.get("schema_linking_sec", 0)

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
                "skip_planner": args.skip_planner,
                "skip_reviewer": args.skip_reviewer,
                "no_few_shot": args.no_few_shot,
                "no_column_meaning": args.no_column_meaning,
                "no_schema_check": args.no_schema_check,
                "online_schema": args.online_schema,
                "index_file": args.index_file,
                "exp_a": args.exp_a,
                "exp_a_codefence": args.exp_a_codefence,
                "exp_b": args.exp_b,
                "exp_b_reversed": args.exp_b_reversed,
                "exp_b_reversed_codefence": args.exp_b_reversed_codefence,
            },
            "latency": latency,
            "plan": plan,
            "sql_tracking": {
                "preliminary_sql": schema_result.preliminary_sql if schema_result else None,
                "initial_sql": initial_sql,
                "revised_sql": revised_sql,
                "final_sql": final_sql,
                "revision_applied": revision_applied,
                "refiner_applied": refiner_applied,
                "verification_applied": verification_applied,
            },
            "schema_linking": {
                "tables": schema_result.tables if schema_result else None,
                "columns": schema_result.columns if schema_result else None,
            } if schema_result else None,
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
