""" Evaluate on Firstloop or BaseMAD """

""" Prompts on the Firstloop """
E:/Anaconda/envs/langchain_env/python.exe Exp/evaluation.py --loop_mode first --dataset spider --split dev
E:/Anaconda/envs/langchain_env/python.exe Exp/evaluation.py --loop_mode first --dataset spider --split dev --max_examples 999999 --planner_temperature 0.0 --skeptic_temperature 0.2 --reasoner_temperature 0.0 --log_path Exp/logs/Qwen2.5-Coder/firstloop_run1_log.json

#Override default paths choose your own paths
E:/Anaconda/envs/langchain_env/python.exe Exp/evaluation.py --loop_mode first --dataset spider --split test --examples_path "Data\spider_data\test.json" --tables_path "Data\spider_data\test_tables.json" --db_root "Data\spider_data\test_database" --gold_sql_path "Data\spider_data\test_gold.sql" --start 0 --max_examples 999999 --planner_temperature 0.0 --skeptic_temperature 0.2 --reasoner_temperature 0.0 --max_tokens 1200
python -m Exp.evaluation.py --loop_mode first --dataset spider --split test --examples_path "Data\spider_data\test.json" --tables_path "Data\spider_data\test_tables.json" --db_root "Data\spider_data\test_database" --gold_sql_path "Data\spider_data\test_gold.sql" --start 0 --max_examples 999999 --planner_temperature 0.0 --skeptic_temperature 0.2 --reasoner_temperature 0.0 --max_tokens 1200

""" Prompts on the BaseMAD """
E:/Anaconda/envs/langchain_env/python.exe Exp/evaluation.py --dataset spider --split dev --loop_mode BaseMAD --start 0 --max_examples 1 --max_tokens 2000 --planner_temperature 0.0 --skeptic_temperature 0.0 --reasoner_temperature 0.0 --max_debate_rounds 3 --min_debate_rounds 1 --log_path Exp/logs/Qwen3/test/BaseMAD_test_log.json


""" Prompts on the ImprovedMAD """
E:/Anaconda/envs/langchain_env/python.exe Exp/evaluation.py --loop_mode ImprovedMAD --dataset spider --split dev --start 0 --max_examples 50 --planner_temperature 0.0 --skeptic_temperature 0.0 --reasoner_temperature 0.0 --max_tokens 2000 --max_debate_rounds 3 --min_debate_rounds 1 --enable_early_termination --log_path Exp/logs/Qwen3/test/ImprovedMad_log.json
