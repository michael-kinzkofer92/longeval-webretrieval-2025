import pandas as pd
from pyserini.search.lucene import LuceneSearcher
import os

# BM25 baseline + traditional model as a class
# based on script/search.py + instructions in Word

class BM25Baseline:
    """
    Class implementing BM25 baseline model
    """
    run_id = 'bm25-baseline+traditional'


    def __init__(self, index_path: str, queries_file_path: str, run_file_path: str):
        self.index_path = index_path
        self.queries_file_path = queries_file_path
        self.run_file_path = run_file_path

    def parse_queries(self) -> pd.DataFrame:
        # Parse queries.trec manually
        queries = []

        with open(self.queries_file_path, 'r') as f:
            qid, query = None, None
            for line in f:
                if line.startswith('<num>'):
                    qid = line.strip()
                    qid = qid.replace('<num>', '').replace('</num>', '')
                    qid = qid.replace('Number:', '').strip()
                if line.startswith('<title>'):
                    query = line.replace('<title>', '').strip()
                    queries.append({'qid': qid, 'query': query})

        return pd.DataFrame(queries)

    def run_search(self, k1, b, top_k):
        # Load searcher
        searcher = LuceneSearcher(self.index_path)

        queries_df = self.parse_queries()

        # Set BM25 parameters (if not set, they stay at k1 = 0.9, b = 0.4)
        searcher.set_bm25(k1=k1, b=b)

        # Create runs/ output directory if missing
        os.makedirs(os.path.dirname(self.run_file_path), exist_ok=True)

        # Write results in TREC format
        with open(self.run_file_path, 'w') as f_out:
            for _, row in queries_df.iterrows():
                qid, query = str(row.qid), row.query
                hits = searcher.search(query, k=top_k)
                for rank, hit in enumerate(hits):
                    docid = hit.docid
                    f_out.write(f"{qid} Q0 {docid} {rank+1} {hit.score:.4f} {self.run_id}\n")

        print(f"✅ Test run written to {self.run_file_path}")