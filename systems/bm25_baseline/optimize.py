import os
import yaml
import subprocess
from bm25_baseline import BM25Baseline

# Define script & project path
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(script_path))

# Load configuration
config_path = os.path.join(script_path, 'bm25_conf.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Define paths from config
INDEX_DIR = str(os.path.join(project_path, config['bm25']['index_dir']))
QUERIES_FILE = str(os.path.join(project_path, config['data']['data_dir'], config['data']['queries_file']))
RUN_FILE = str(os.path.join(project_path, config['general']['output_dir'], 'run_bm25.txt'))

# BM25 parameters from config

# Controls term frequency scaling
k1_range = config['bm25']['k1 range']
# Controls document length normalization
b_range = config['bm25']['b range']
# Controls how many documents are returned for each query
top_k = config['bm25'].get('top_k', 25)

# BM25 search as an object
BM25 = BM25Baseline(index_path=INDEX_DIR,
                    queries_file_path=QUERIES_FILE,
                    run_file_path=RUN_FILE)

# Defining path for optimization
eval_path = os.path.join(project_path, config["optimization"]["evaluate path"])
qrels_path = os.path.join(project_path, config["optimization"]["qrels path"])
run_path = os.path.join(project_path, config["optimization"]["run path"])
results_path = os.path.join(project_path, config["optimization"]["results path"])

evaluation_config_map = {}
index = 0
# Trying every parameter permutation
for k in k1_range:
    for b in b_range:
        # Search
        BM25.run_search(k1=k, b=b, top_k=top_k)

        # File name and parameters map
        file_name = f"opt_{index}.txt"
        index += 1
        evaluation_config_map[file_name] = {"k": k, "b": b}

        # Evaluate using scripts/evaluate.py
        evaluation_completed = subprocess.run([".venv/bin/python3", 
                                               eval_path, 
                                               "--qrels", qrels_path, 
                                               "--run", run_path, 
                                               "--output", os.path.join(results_path, "result1")])
        break

# Compare eval results & try to find the best combination
best_result = 0.0
best_config = None
for file_name in evaluation_config_map:
    with open(os.path.join(results_path, file_name), "r") as file:
        for line in file:
            if line.startswith("Average"):
                average_line = line.strip().split(" = ")
                if float(average_line[1]) > best_result:
                    best_result = float(average_line[1])
                    best_config = evaluation_config_map[file_name]

# Save best parameter to config
if best_config is not None:
    config["optimization"]["optimized k"] = best_config["k"]
    config["optimization"]["optimized b"] = best_config["b"]

with open(config_path, "w") as yaml_config:
    yaml.dump(config, yaml_config)