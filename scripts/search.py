import pandas as pd
from pyserini.search.lucene import LuceneSearcher
import os

# Paths
INDEX_DIR = 'index/bm25'
QUERIES_FILE = 'data/queries.tsv'
RUN_FILE = 'runs/run_bm25.txt'
RUN_ID = 'bm25-baseline'

# Load searcher
searcher = LuceneSearcher(INDEX_DIR)

# Load queries
queries = pd.read_csv(QUERIES_FILE, sep='\t', header=None, names=['qid', 'query'], dtype={'qid': str})

# Create runs/ if missing
os.makedirs(os.path.dirname(RUN_FILE), exist_ok=True)

# Write results in TREC format
with open(RUN_FILE, 'w') as f_out:
    for _, row in queries.iterrows():
        qid, query = str(row.qid), row.query
        hits = searcher.search(query, k=1000)
        for rank, hit in enumerate(hits):
            f_out.write(f"{qid} Q0 {hit.docid} {rank + 1} {hit.score:.4f} {RUN_ID}\n")

print(f"✅ Run written to {RUN_FILE}")
