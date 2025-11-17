#Evaluate the Planner baseline on the Spider test set
E:/Anaconda/envs/langchain_env/python.exe baseline/evaluation.py  --baseline planner --split test --max_examples 999999
python -m baseline.evaluation.py --baseline planner --split test --max_examples 999999

#Evaluate the QwenAgent baseline on the Spider dev set
E:/Anaconda/envs/langchain_env/python.exe baseline/evaluation.py  --baseline baseagent --split dev --max_examples 999999
python -m baseline.evaluation.py --baseline baseagent --split dev --max_examples 999999

#Override default paths choose your own paths
E:/Anaconda/envs/langchain_env/python.exe baseline/evaluation.py --baseline planner --split test --examples_path "Data\spider_data\test.json" --tables_path "Data\spider_data\test_tables.json" --db_root "Data\spider_data\test_database" --gold_sql_path "Data\spider_data\test_gold.sql" --start 0 --max_examples 999999 --temperature 0.0 --max_tokens 1200
python -m baseline.evaluation.py -baseline planner --split test --examples_path "Data\spider_data\test.json" --tables_path "Data\spider_data\test_tables.json" --db_root "Data\spider_data\test_database" --gold_sql_path "Data\spider_data\test_gold.sql" --start 0 --max_examples 999999 --temperature 0.0 --max_tokens 1200