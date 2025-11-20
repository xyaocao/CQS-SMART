''' There are two types of inputs: 
1. Single run inputs: Provide one question and a correspoinding db_id.
2. Batch run inputs: Provide a file containing multiple questions and their corresponding db_ids in JSONL format.
'''
#!/bin/bash
# Type 1: Single run
python -m Exp.run_multiagent_firstloop "question" "db_id" --dataset spider --input_mode single
E:/Anaconda/envs/langchain_env/python.exe Exp/run_multiagent_firstloop.py "question" "db_id" --dataset spider --input_mode single

# Type 2: Batch run
python -m bExp.run_multiagent_firstloop --dataset spider --split dev --input_mode batch --start 10 --num_examples 5
E:/Anaconda/envs/langchain_env/python.exe Exp/run_multiagent_firstloop.py --dataset spider --split dev --input_mode batch --start 10 --num_examples 5
