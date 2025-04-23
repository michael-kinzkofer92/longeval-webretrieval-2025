import os
import subprocess

CORPUS_DIR = 'data/corpus'
INDEX_DIR = 'index/bm25'

# Create an output directory if needed
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



