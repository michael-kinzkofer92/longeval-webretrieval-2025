import os
import subprocess
import yaml

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Get paths from config
CORPUS_DIR = os.path.join(
    config['data']['data_dir'],
    config['data']['train_collection'],
    'Json/2022-06_fr'
)
INDEX_DIR = config['bm25']['index_dir']

# Create output directory if needed
os.makedirs(INDEX_DIR, exist_ok=True)

# Build the command
command = [
    'python', '-m', 'pyserini.index.lucene',
    '--collection', 'JsonCollection',
    '--input', CORPUS_DIR,
    '--index', INDEX_DIR,
    '--generator', 'DefaultLuceneDocumentGenerator',
    '--threads', '2',
    '--storePositions',
    '--storeDocvectors',
    '--storeRaw'
]

print("🔨 Start indexing...")

# Execute the command
try:
    subprocess.run(command, check=True)
    print(f"✅ Index successfully created at: {INDEX_DIR}")
except subprocess.CalledProcessError as e:
    print("❌ Error during indexing:")
    print(e)



