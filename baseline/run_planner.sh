''' There are two types of inputs: 
1. Single run inputs: Provide one question and a correspoinding db_id.
2. Batch run inputs: Provide a file containing multiple questions and their corresponding db_ids in JSONL format.
'''
#!/bin/bash
# Type 1: Single run
python -m baseline.run_planner "question" "db_id" --dataset spider --input_mode single
E:/Anaconda/envs/langchain_env/python.exe baseline/run_planner.py "question" "db_id" --dataset spider --input_mode single

# Type 2: Batch run
python -m baseline.run_planner --dataset spider --split dev --input_mode batch --start 10 --num_examples 5
E:/Anaconda/envs/langchain_env/python.exe baseline/run_planner.py --dataset spider --split dev --input_mode batch --start 10 --num_examples 5
