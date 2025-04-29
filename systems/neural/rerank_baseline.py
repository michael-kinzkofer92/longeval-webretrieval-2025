import argparse
import json, os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# --------------------------------------------------
# Konfiguration
# --------------------------------------------------
BM25_RUN_FILE   = 'runs/run_bm25.txt'                 # Top-100-Listen aus Pyserini
DOCUMENT_CORPUS = 'data/corpus/corpus.jsonl'          # Volltexte
QUERIES_FILE    = 'data/queries.tsv'                  # <qid>\t<query>
OUTPUT_RUN_FILE = 'runs/run_neural_monoT5.txt'
MODEL_NAME      = 'castorini/monot5-base-msmarco'     # Cross-Encoder-Variante
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K = 100                                           # Tiefe des Rerankings
BATCH_SIZE = 8                                        # GPU-Speicherschonend

# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def load_corpus(path):
    corpus = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            corpus[str(obj['id'])] = obj['contents']
    return corpus

def load_run(path, k=TOP_K):
    runs = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rank, *_ = line.split()
            runs.setdefault(qid, [])
            if len(runs[qid]) < k:                    # nur Top-k mitnehmen
                runs[qid].append(docid)
    return runs

def rerank(model, tokenizer, query, docs, batch_size=BATCH_SIZE):
    scores = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_inputs = [
            f"Query: {query} Document: {d}" for d in batch_docs
        ]
        enc = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits               # (B, num_labels)
            # wir nehmen den Logit des "relevant"-Labels an Index 1
            if logits.size(-1) > 1:
                logits = logits[:, 1]
            else:                                      # 1-Label-Kopf
                logits = logits.squeeze(-1)
        scores.extend(logits.cpu().tolist())           # Liste von Floats
    return scores

# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------
def main():
    print("⏳ Lade Daten …")
    corpus   = load_corpus(DOCUMENT_CORPUS)
    bm25_run = load_run(BM25_RUN_FILE)
    queries  = pd.read_csv(QUERIES_FILE, sep='\t', header=None,
                           names=['qid', 'query'], dtype=str)\
                 .set_index('qid')['query'].to_dict()

    print("⏳ Lade Modell …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)\
                 .to(DEVICE).eval()

    os.makedirs(os.path.dirname(OUTPUT_RUN_FILE), exist_ok=True)

    print("🚀 Starte Re-Ranking …")
    with open(OUTPUT_RUN_FILE, 'w') as fout:
        for qid, docids in tqdm(bm25_run.items()):
            if qid not in queries:
                continue                              # Sicherheit
            query = queries[qid]

            # 1) nur Dokumente, die wir wirklich haben
            pairs = [(d, corpus[d]) for d in docids if d in corpus]
            if not pairs:
                continue
            docids_filtered, texts = zip(*pairs)

            # 2) Scoren
            scores = rerank(model, tokenizer, query, texts)

            # 3) Sortieren & Ausgeben
            ranked = sorted(zip(docids_filtered, scores),
                           key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, 1):
                fout.write(f"{qid} Q0 {docid} {rank} {score:.4f} monoT5\n")

    print(f"✅ Fertig! Ergebnis unter {OUTPUT_RUN_FILE}")

if __name__ == "__main__":
    main()
