''' There are two types of inputs: 
1. Single run inputs: Provide one question and a correspoinding db_id.
2. Batch run inputs: Provide a file containing multiple questions and their corresponding db_ids in JSONL format.
'''
#!/bin/bash
# Type 1: Single run
python -m Exp.run_BaseMAD "question" "db_id" --dataset spidert --split dev --loop_mode BaseMAD --input_mode single --max_debate_rounds 3 --min_debate_rounds 2 
E:/Anaconda/envs/langchain_env/python.exe Exp/run_BaseMAD.py "question" "db_id" --loop_mode BaseMAD --dataset spider --split dev --input_mode single --max_debate_rounds 3 --min_debate_rounds 2 

# Type 2: Batch run
python -m Exp.run_BaseMAD --dataset spider --split dev --input_mode batch --start 10 --num_examples 5 --max_debate_rounds 3 --min_debate_rounds 2 
E:/Anaconda/envs/langchain_env/python.exe Exp/run_BaseMAD.py --dataset spider --split dev --input_mode batch --start 10 --num_examples 5 --max_debate_rounds 3 --min_debate_rounds 2 