#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light-weight HuggingFace mono-BERT reranker
für LongEval WebRetrieval – TESTDATEN (TREC / JSON).

Input  : web-submission-traditional/<month>/run.txt.gz – Top-25 Dokumente
Output : web-submission-neuronal/<month>/run.txt       – rerankte Liste
"""

from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import re, gzip, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import argparse


# ---------------------------------------------- #
ROOT     = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "data/test/LongEval Test Collection"
RUN_ROOT  = ROOT / "web-submission-traditional"
OUT_ROOT  = ROOT / "web-submission-neuronal"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 1024
TOP_K      = 10

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

DOC_START = re.compile(r"<DOC>")
DOC_END   = re.compile(r"</DOC>")
DOCNO     = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.I)


# ---------------- GPU Device Setup + Check ---------------- #
import torch

FORCE_GPU = True
USE_FP16 = False

if FORCE_GPU:
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA requested but not available!")
    DEVICE = torch.device("cuda")
    USE_FP16 = True
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_FP16 = DEVICE.type == "cuda"

print(f"📦 DEVICE = {DEVICE}  |  FP16 = {USE_FP16}")


# ----------------------- Loaders ------------------------ #
def read_run(path: Path, k: int = TOP_K) -> Dict[str, List[str]]:
    run = {}
    with smart_open(path) as f:
        for ln in f:
            qid, _, docid, *_ = ln.strip().split()
            run.setdefault(qid, [])
            if len(run[qid]) < k:
                run[qid].append(docid)
    return run


def read_queries_txt(path: Path) -> Dict[str, str]:
    q = {}
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                qid, txt = ln.rstrip("\n").split("\t", 1)
                q[qid] = txt
    return q

def chunk_text(text: str, chunk_size: int = 250) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def parse_trec_file(fp: Path, wanted: Set[str]) -> Dict[str, str]:
    corpus = {}
    with smart_open(fp) as f:
        in_doc, buf, docid = False, [], None
        for ln in f:
            if DOC_START.match(ln):
                in_doc, buf, docid = True, [], None; continue
            if DOC_END.match(ln):
                in_doc = False
                if docid and docid in wanted:
                    chunks = chunk_text(" ".join(buf), 100)
                    for i, chunk in enumerate(chunks or [" ".join(buf)]):
                        corpus[f"{docid}-{i}"] = chunk
                continue
            if in_doc:
                if docid is None and (m := DOCNO.search(ln)):
                    docid = m.group(1).strip()
                else:
                    buf.append(ln.strip())
    return corpus

# ------------------------ Rerank ------------------------ #
def rerank(model, tok, query: str, docs: List[str]) -> List[float]:
    scores = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        enc = tok(
            [f"How relevant is the following document to the query?\nQuery: {query}\nDocument: {d}" for d in batch],
            padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            if USE_FP16:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits
        logits = logits[:, 1] if logits.size(-1) > 1 else logits.squeeze(-1)
        scores.extend(logits.float().cpu().tolist())
    return scores


from pathlib import Path
import gzip

def smart_open(path: Path):
    with path.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    else:
        return open(path, "rt", encoding="utf-8", errors="ignore")



# ------------------------ Main Loop --------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", required=True, help="Liste von Monaten z. B. 2023-03 2023-04")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # Prüfe tatsächlich verwendetes Gerät
    actual_device = next(model.parameters()).device
    print(f"📦 DEVICE = {DEVICE}  |  FP16 = {USE_FP16}")
    print("📊 Model loaded on:", actual_device)
    if actual_device.type != "cuda":
        print("⚠️  WARNUNG: Modell läuft NICHT auf der GPU – bitte prüfen!")

    for month in args.months:
        print(f"\n================ {month} ================")
        run_fp  = RUN_ROOT / month / "run.txt.gz"
        out_dir = OUT_ROOT / month; out_dir.mkdir(parents=True, exist_ok=True)
        out_fp  = out_dir / "run.txt"
        query_fp = TEST_ROOT / "queries" / f"{month}_queries.txt"
        trec_dir = TEST_ROOT / "Trec" / f"{month}_fr" / "collection"

        if not run_fp.exists():
            print("✖ Runfile fehlt:", run_fp)
            continue
        if not query_fp.exists():
            print("✖ Queries fehlen:", query_fp)
            continue

        # Schritt 1: Laden
        bm25 = read_run(run_fp)
        queries = read_queries_txt(query_fp)

        # Schritt 2: Subsampling
        import random
        random.seed(42)
        subset_qids = random.sample(list(bm25.keys()), 2500)
        bm25 = {qid: bm25[qid] for qid in subset_qids}
        queries = {qid: queries[qid] for qid in subset_qids}

        needed = {d for lst in bm25.values() for d in lst}
        docs = {}
        for fp in trec_dir.glob("*.jsonl.gz"):
            docs.update(parse_trec_file(fp, needed))
        print(f"✅ Loaded {len(docs)} passages for {len(set(k.split('-')[0] for k in docs))} docs")

        with out_fp.open("w") as fout:
            for qid, docids in tqdm(bm25.items(), desc=f"⚡ Reranking {month}"):
                if qid not in queries:
                    continue
                chunks, ids = [], []
                for docid in docids:
                    hits = [k for k in docs if k.startswith(docid + "-")]
                    if not hits:
                        continue
                    chunks.extend(docs[k] for k in hits)
                    ids.extend(hits)

                if not chunks:
                    continue

                scores = rerank(model, tok, queries[qid], chunks)
                merged = defaultdict(list)
                for cid, sc in zip(ids, scores):
                    merged[cid.split("-")[0]].append(sc)
                ranked = sorted(
                    ((doc, sum(vals) / len(vals)) for doc, vals in merged.items()),
                    key=lambda x: x[1], reverse=True
                )
                for rank, (doc, score) in enumerate(ranked, 1):
                    fout.write(f"{qid} Q0 {doc} {rank} {score:.4f} luyuHF\n")
        print("🏁 Fertig →", out_fp)


if __name__ == "__main__":
    main()
