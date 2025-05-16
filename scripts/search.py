import pandas as pd
from pyserini.search.lucene import LuceneSearcher
import os
import yaml

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paths from config
INDEX_DIR = config['bm25']['index_dir']
QUERIES_FILE = os.path.join(config['data']['data_dir'], config['data']['queries_file'])
RUN_FILE = os.path.join(config['general']['output_dir'], 'run_bm25.txt')
RUN_ID = 'bm25-baseline'

# BM25 parameters from config
k1 = config['bm25'].get('k1', 0.9)
b = config['bm25'].get('b', 0.4)
top_k = config['bm25'].get('top_k', 25)

# Load searcher
searcher = LuceneSearcher(INDEX_DIR)

# Parse queries.trec manually
queries = []
with open(QUERIES_FILE, 'r') as f:
    qid, query = None, None
    for line in f:
        if line.startswith('<num>'):
            qid = line.strip()
            qid = qid.replace('<num>', '').replace('</num>', '')
            qid = qid.replace('Number:', '').strip()
        if line.startswith('<title>'):
            query = line.replace('<title>', '').strip()
            queries.append({'qid': qid, 'query': query})
queries_df = pd.DataFrame(queries)

# downsampling to 1000 queries for testing
qrels_qids = set()

QRELS_FILE = os.path.join(config['data']['data_dir'], 'LongEval Train Collection/qrels/2022-11_fr/qrels_processed.txt')
with open(QRELS_FILE, 'r') as f:
    for line in f:
        qid, *_ = line.strip().split()
        qrels_qids.add(qid)

queries_df = queries_df[queries_df['qid'].isin(qrels_qids)]
queries_df = queries_df.sample(n=1000, random_state=42)

# Create runs/ if missing
os.makedirs(os.path.dirname(RUN_FILE), exist_ok=True)

# Write results in TREC format
with open(RUN_FILE, 'w') as f_out:
    for _, row in queries_df.iterrows():
        qid, query = str(row.qid), row.query
        hits = searcher.search(query, k=top_k)
        print(f"Query {qid} → Top hits:", [hit.docid for hit in hits[:10]])
        for rank, hit in enumerate(hits):
            docid = hit.docid
            if docid.startswith("doc"):
                docid = docid[3:]  # strip "doc" prefix
            f_out.write(f"{qid} Q0 {docid} {rank+1} {hit.score:.4f} {RUN_ID}\n")



print(f"✅ Test run written to {RUN_FILE}")
