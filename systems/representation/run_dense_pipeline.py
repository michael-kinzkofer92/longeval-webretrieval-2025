#!/usr/bin/env python3
"""
Generic dense‑retrieval pipeline for LongEval (French Web) – SBERT → FAISS → TREC run + optional evaluation.

Usage (example – variant 02 from the test‑plan):
    python systems/representation/run_dense_pipeline.py \
        --run-id 02 \
        --model distiluse-base-multilingual-cased-v1 \
        --embedding mean \
        --faiss flat \
        --topk 1000 \
        --evaluate   # ← runs evaluate.py for Lag‑6 & Lag‑8 automatically

CLI flags
---------
--run-id        short identifier used for output files (e.g. "02")
--model         HuggingFace model name or local path
--embedding     cls | mean               (default: cls)
--faiss         flat | hnsw             (default: flat)
--dim-reduce    integer (e.g. 128)      (optional PCA dimensionality)
--topk          documents per query      (default 1000)
--batch-size    (embedding batch, default 64)
--evaluate      if set, call scripts/evaluate.py afterwards

Outputs
-------
runs/run_dense_<run-id>.txt              – TREC runfile
faiss_indices/index_<run-id>.faiss       – optional saved FAISS index
Evaluation results stored in eval_results/ (only if --evaluate)

Requirements
------------
* sentence‑transformers
* faiss‑cpu   (or faiss‑gpu if available)
* tqdm, yaml

This script is self‑contained and can be reused for all 8 variants by 
changing CLI flags or looping over a list.
"""

from __future__ import annotations
import argparse, json, os, re, sys, time, pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import yaml
import torch
import faiss
faiss.omp_set_num_threads(os.cpu_count())



from sentence_transformers import SentenceTransformer, models

# ------------------------------------------------------------
# Paths & constants (inherit from project config.yml)
# ------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]       # project root
CFG_PATH    = ROOT / "scripts" / "config.yml"
cfg = yaml.safe_load(Path(CFG_PATH).read_text())

DATA_DIR    = ROOT / cfg["data"]["data_dir"]
OUTPUT_DIR  = ROOT / cfg["general"]["output_dir"]

TREC_CORPUS_DIR = ROOT / "data/lag6_lag8_subset/release_2025_p1/French/LongEval Train Collection/Trec"
BM25_RUN_FILE   = ROOT / "runs/run_bm25.txt"
QUERIES_TREC    = ROOT / "data/lag6_lag8_subset/release_2025_p1/French/queries.trec"
QRELS_LAG6      = ROOT / "data/lag6_lag8_subset/release_2025_p1/French/LongEval Train Collection/qrels/2022-11_fr/qrels_processed.txt"
QRELS_LAG8      = ROOT / "data/lag6_lag8_subset/release_2025_p1/French/LongEval Train Collection/qrels/2023-01_fr/qrels_processed.txt"

# ------------------------------------------------------------
# TREC helpers (reuse from BM25 scripts)
# ------------------------------------------------------------
DOC_START = re.compile(r"<DOC>")
DOC_END   = re.compile(r"</DOC>")
DOCNO     = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.I)
TITLE_TAG = re.compile(r"<TITLE>(.*?)</TITLE>", re.I)


def parse_queries_trec(path: Path) -> Dict[str, str]:
    mapping, qid = {}, None
    for ln in path.read_text("utf-8").splitlines():
        if ln.startswith("<num>"):
            qid = (ln.replace("<num>", "")
                     .replace("</num>", "")
                     .replace("Number:", "").strip())
        elif ln.startswith("<title>"):
            mapping[qid] = ln.replace("<title>", "").strip()
    return mapping


CORPUS_CACHE = ROOT / "embeddings/corpus_cache.pkl"

def load_corpus_texts(directory: Path) -> Dict[str, str]:
    if CORPUS_CACHE.exists():
        print(f"📂 Loading cached corpus from {CORPUS_CACHE.name}")
        with CORPUS_CACHE.open("rb") as f:
            return pickle.load(f)

    print("📑 Parsing corpus from TREC files …")
    corpus: Dict[str, str] = {}
    for fp in tqdm(directory.rglob("*.trec"), desc="Reading corpus"):
        with fp.open("r", encoding="utf-8") as f:
            in_doc, buf, docid = False, [], None
            for ln in f:
                if DOC_START.match(ln):
                    in_doc, buf, docid = True, [], None
                    continue
                if in_doc and DOC_END.match(ln):
                    in_doc = False
                    if docid:
                        corpus[docid] = " ".join(buf)
                    continue
                if in_doc:
                    if docid is None and (m := DOCNO.search(ln)):
                        docid = m.group(1).strip()
                    else:
                        buf.append(ln.strip())
    # 🧊 Save to cache
    with CORPUS_CACHE.open("wb") as f:
        pickle.dump(corpus, f)
    print(f"✅ Cached corpus to {CORPUS_CACHE.name}")
    return corpus


# ------------------------------------------------------------
# Embedding utilities
# ------------------------------------------------------------

def embed_texts(model: SentenceTransformer, texts: List[str], batch: int = 512,
                strategy: str = "cls") -> np.ndarray:
    """Return matrix (n, dim) using CLS or mean pooling."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch), desc="Embedding", unit="batch"):
        batch_txt = texts[i:i+batch]
        embs = model.encode(batch_txt, convert_to_numpy=True, show_progress_bar=False)
        if strategy == "cls":
            # sentence_transformers CLS pooler already returns CLS if model has it
            embeddings.append(embs)
        else:
            embeddings.append(embs)
    return np.vstack(embeddings)

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def build_faiss(dim: int, index_type: str = "flat") -> faiss.Index:
    if index_type == "flat":
        idx = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        idx = faiss.IndexHNSWFlat(dim, 32)
    else:
        raise ValueError("faiss-type must be flat or hnsw")

    if faiss.get_num_gpus() > 0:
        print(f"🚀 FAISS GPU detected → using GPU index for {index_type.upper()} with dim {dim}")
        res = faiss.StandardGpuResources()
        idx = faiss.index_cpu_to_gpu(res, 0, idx)

    return idx



def run_pipeline(args):
    run_tag = f"dense_{args.run_id}"
    emb_file = ROOT / f"embeddings/doc_embeddings_{run_tag}.npy"
    run_file = ROOT / f"runs/run_{run_tag}.txt"
    run_file.parent.mkdir(parents=True, exist_ok=True)
    idx_file = ROOT / f"faiss_indices/index_{run_tag}.faiss"

    # 1) Load corpus texts once
    print("📚 Loading corpus …")
    corpus = load_corpus_texts(TREC_CORPUS_DIR)
    docids = list(corpus.keys())
    texts  = list(corpus.values())
    print(f"→ {len(docids):,} docs loaded")

    # 2) SBERT model & encoding
    print(f"🔤 Loading model {args.model} …")
    model = SentenceTransformer(args.model, device=DEVICE)
    dim   = model.get_sentence_embedding_dimension()
    
    # Optionaler Pfad für Zwischenspeicherung
    emb_file = ROOT / f"embeddings/doc_embeddings_{run_tag}.npy"
    pca_file = ROOT / f"embeddings/pca_{run_tag}.pkl"

    
    #if emb_file.exists():
     #   print(f"📦 Embeddings cached → loading from {emb_file.name}")
      #  doc_embeddings = np.load(emb_file)
       # if args.dim_reduce and pca_file.exists():
        #    print(f"⚙️ PCA transform cached → loading from {pca_file.name}")
         #   with open(pca_file, "rb") as f:
          #      pca = pickle.load(f)
           #     doc_embeddings = pca.transform(doc_embeddings)
            #    dim = args.dim_reduce


    if emb_file.exists():
        print(f"📦 Embeddings cached → loading from {emb_file.name}")
        doc_embeddings = np.load(emb_file)
        if args.dim_reduce and pca_file.exists():
            print(f"⚙️ PCA transform cached → skipping transform (already reduced)")
        dim = args.dim_reduce if args.dim_reduce else doc_embeddings.shape[1]

    else:
        print("🧮 Embedding documents …")
        doc_embeddings = embed_texts(model, texts, args.batch_size, args.embedding)
        if args.dim_reduce and args.dim_reduce < dim:
            from sklearn.decomposition import PCA
            print(f"🔻 Reducing dimension to {args.dim_reduce} with PCA …")
            pca = PCA(n_components=args.dim_reduce, random_state=42)
            doc_embeddings = pca.fit_transform(doc_embeddings)
            with open(pca_file, "wb") as f:
                pickle.dump(pca, f)
            dim = args.dim_reduce
        np.save(emb_file, doc_embeddings)
        print(f"✅ Saved embeddings to {emb_file.name}")



    # 3) Build FAISS index but only reuse cache if embedding shape matches
    if idx_file.exists():
        index = faiss.read_index(str(idx_file))
        if index.ntotal != doc_embeddings.shape[0]:
            print(f"⚠️ Index mismatch (has {index.ntotal}, but embeddings have {doc_embeddings.shape[0]}) → rebuilding index")
            index = build_faiss(dim, args.faiss)
            index.add(doc_embeddings.astype("float32"))
            faiss.write_index(index, str(idx_file))
            print(f"✅ FAISS index rebuilt and written to {idx_file}")
        else:
            print(f"📦 FAISS index cached → loading from {idx_file.name}")
    else:
        print("⚙️ Building FAISS index …")
        index = build_faiss(dim, args.faiss)
        index.add(doc_embeddings.astype("float32"))
        idx_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(idx_file))
        print(f"✅ FAISS index written to {idx_file}")



    
    print(f"✅ FAISS index written to {idx_file}")

    # 4) Encode queries
    queries = parse_queries_trec(QUERIES_TREC)

    max_queries = 300
    qids, qtexts = zip(*list(queries.items())[:max_queries])

    qids = qids[:max_queries]
    qtexts = qtexts[:max_queries]

    
    query_emb_file = ROOT / f"embeddings/query_embeddings_{run_tag}.npy"
    
    if query_emb_file.exists():
        print(f"📦 Query embeddings cached → loading from {query_emb_file.name}")
        query_emb = np.load(query_emb_file)
        if args.dim_reduce and args.dim_reduce < dim:
            query_emb = pca.transform(query_emb)
    else:
        print("🧮 Embedding queries …")
        query_emb = embed_texts(model, list(qtexts), args.batch_size, args.embedding)
        if args.dim_reduce and args.dim_reduce < dim:
            query_emb = pca.transform(query_emb)
        np.save(query_emb_file, query_emb)
        print(f"✅ Saved query embeddings to {query_emb_file.name}")







    # 5) Search
    print("🔎 Retrieval …")
    sim_list, idx_list = [], []
    print("🔎 Retrieval (batched) …")
    for i in tqdm(range(0, len(query_emb), 16), desc="Retrieving"):
        q_batch = query_emb[i:i+16]
        sim_batch, idx_batch = index.search(q_batch.astype("float32"), args.topk)
        sim_list.append(sim_batch)
        idx_list.append(idx_batch)
    
    sim = np.vstack(sim_list)
    idxs = np.vstack(idx_list)


    # 6) Write TREC run file
    with run_file.open("w") as fout:
        for qi, qid in enumerate(qids):
            for rank, (doc_idx, score) in enumerate(zip(idxs[qi], sim[qi]), 1):
                fout.write(f"{qid} Q0 {docids[doc_idx].lstrip('doc')} {rank} {score:.4f} {run_tag}\n")
    print(f"🏁 Run written → {run_file}")

    # 7) Optional evaluation
    if args.evaluate:
        eval6 = ROOT / f"eval_results/eval_{run_tag}_lag6.txt"
        eval8 = ROOT / f"eval_results/eval_{run_tag}_lag8.txt"
        drop  = ROOT / f"eval_results/eval_{run_tag}_drop.txt"
        cmd6  = f"python scripts/evaluate.py --qrels '{QRELS_LAG6}' --run '{run_file}' --output '{eval6}'"
        cmd8  = f"python scripts/evaluate.py --qrels '{QRELS_LAG8}' --run '{run_file}' --output '{eval8}'"
        cmp   = f"python scripts/compare_eval.py --lag6 {eval6} --lag8 {eval8} --output {drop}"
        print(cmd6); os.system(cmd6)
        print(cmd8); os.system(cmd8)
        print(cmp ); os.system(cmp)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense SBERT→FAISS pipeline for LongEval")
    parser.add_argument("--run-id", required=True, help="short id, e.g. 01, 02 …")
    parser.add_argument("--model",   required=True, help="HuggingFace model name")
    parser.add_argument("--embedding", choices=["cls", "mean"], default="cls")
    parser.add_argument("--faiss",    choices=["flat", "hnsw"], default="flat")
    parser.add_argument("--dim-reduce", type=int, default=None, help="PCA dim (< original)")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    run_pipeline(args)
