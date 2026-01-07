car_1: 87–178 (92 examples) → e.g. --start 87 --max_examples 40 for a quick slice.
flight_2: 179–258 (80) → e.g. --start 179 --max_examples 40.
cre_Doc_Template_Mgt: 297–380 (84) → e.g. --start 297 --max_examples 40.
wta_1: 429–490 (62) → e.g. --start 429 --max_examples 40.
student_transcripts_tracking: 507–584 (78) → e.g. --start 507 --max_examples 40.
tvshow: 585–646 (62) → e.g. --start 585 --max_examples 40.
world_1: 702–821 (120) → e.g. --start 702 --max_examples 50.
orchestra: 822–861 (40) → e.g. --start 822 --max_examples 40.
network_1: 862–917 (56) → e.g. --start 862 --max_examples 40.
dog_kennels: 918–999 (82) → e.g. --start 918 --max_examples 40.

E:/Anaconda/envs/langchain_env/python.exe Exp/evaluation.py --loop_mode ImprovedMAD --start 87 --max_examples 92 --dataset spider --split dev --planner_temperature 0.0 --skeptic_temperature 0.0 --reasoner_temperature 0.0 --max_tokens 2000 --max_debate_rounds 3 --min_debate_rounds 1 --enable_early_termination --log Exp/logs/Qwen2.5-Coder/test/improvedmad_new_car_1.json