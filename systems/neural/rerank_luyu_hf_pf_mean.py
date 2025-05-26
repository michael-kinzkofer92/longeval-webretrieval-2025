#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light-weight HuggingFace ‘Luyu-20-w06’-Style mono-BERT reranker
für LongEval WebRetrieval (kein PyGaggle nötig).

Input  : runs/run_bm25.txt      – Top-25 Dokumente pro Query
Output : runs/run_neural_luyu_prompt_clarity_mean.txt  (TREC-Format)

"""

from pathlib import Path
from typing import Dict, List, Set
import re, yaml, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

# ------------------------------------------------------------------------- #
# Config                                                                    #
# ------------------------------------------------------------------------- #
CFG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "config.yml"
cfg = yaml.safe_load(CFG_PATH.read_text())

DATA_DIR    = Path(cfg["data"]["data_dir"])
OUTPUT_DIR  = Path(cfg["general"]["output_dir"])
DOCUMENT_DIR = Path(
    "data/lag6_lag8_subset/release_2025_p1/"
    "French/LongEval Train Collection/Trec"     
)

BM25_RUN = Path("runs/run_bm25.txt")
OUT_FILE = Path("runs/run_neural_luyu_prompt_clarity_mean_topk50.txt")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K      = 50
BATCH_SIZE = 128

DEVICE  = "cuda" if torch.cuda.is_available() else (
          "mps"  if torch.backends.mps.is_available() else "cpu")
USE_FP16 = DEVICE == "cuda"

# ------------------------------------------------------------------------- #
# Helpers                                                                   #
# ------------------------------------------------------------------------- #
DOC_START = re.compile(r"<DOC>")
DOC_END   = re.compile(r"</DOC>")
DOCNO     = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.I)

def load_run(path: Path, k: int = TOP_K) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for ln in path.read_text().splitlines():
        qid, _, docid, *_ = ln.split()
        run.setdefault(qid, [])
        if len(run[qid]) < k:
            run[qid].append(docid.strip())
    return run

def parse_queries_trec(trec: Path) -> Dict[str, str]:
    mapping, qid = {}, None
    for ln in trec.read_text(encoding="utf-8").splitlines():
        if ln.startswith("<num>"):
            qid = ln.replace("<num>", "").replace("</num>", "") \
                   .replace("Number:", "").strip()
        elif ln.startswith("<title>"):
            mapping[qid] = ln.replace("<title>", "").strip()
    return mapping

def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def load_docs(directory: Path, needed: Set[str]) -> Dict[str, str]:
    """Lädt alle .trec-Dateien, splittet in ~100-Token-Chunks."""
    plain_needed = {d.lstrip("doc") for d in needed}
    corpus: Dict[str, str] = {}

    for fp in tqdm(directory.rglob("*.trec"), desc="📖 Scanning & chunking"):
        with fp.open(encoding="utf-8") as f:
            in_doc, buf, docid = False, [], None
            for ln in f:
                if DOC_START.match(ln):
                    in_doc, buf, docid = True, [], None
                    continue
                if in_doc and DOC_END.match(ln):
                    in_doc = False
                    if docid and docid.lstrip("doc") in plain_needed:
                        full_text = " ".join(buf)
                        chunks = chunk_text(full_text, 100) or [full_text]
                        num = docid.lstrip("doc")            # nur Zahl
                        for i, chunk in enumerate(chunks):
                            corpus[f"doc{num}-{i}"] = chunk   # mit Präfix
                            corpus[f"{num}-{i}"]    = chunk   # ohne Präfix
                    continue
                if in_doc:
                    if docid is None and (m := DOCNO.search(ln)):
                        docid = m.group(1).strip()
                    else:
                        buf.append(ln.strip())
    print(f"✅ Loaded {len(corpus)} passages for {len(set(k.split('-')[0] for k in corpus))} docs")
    return corpus

def rerank(model, tok, query: str, docs: List[str]) -> List[float]:
    scores: List[float] = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        enc = tok(
            [f"How relevant is the following document to the query?\n"
             f"Query: {query}\nDocument: {d}"
             for d in batch],
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

# ------------------------------------------------------------------------- #
# Main                                                                      #
# ------------------------------------------------------------------------- #
def main() -> None:
    bm25   = load_run(BM25_RUN)
    needed = {d for lst in bm25.values() for d in lst}
    docs   = load_docs(DOCUMENT_DIR, needed)
    queries = parse_queries_trec(
        Path("data/lag6_lag8_subset/release_2025_p1/French/queries.trec")
    )

    print(f"⏳ Loading {MODEL_NAME} on {DEVICE} …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = (AutoModelForSequenceClassification
             .from_pretrained(MODEL_NAME,
                              torch_dtype=torch.float16 if USE_FP16 else None)
             .to(DEVICE).eval())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as fout:
        for qid, docids in tqdm(bm25.items(), desc="⚡ Reranking"):
            if qid not in queries:
                continue

            texts, chunk_ids = [], []
            for docid in docids:                       
                num = docid.lstrip("doc")
                chunks = [k for k in docs
                          if k.startswith(num + "-") or k.startswith("doc"+num + "-")]
                if not chunks:
                    continue
                texts.extend(docs[k] for k in chunks)
                chunk_ids.extend(chunks)

            if not texts:
                continue

            scores = rerank(model, tok, queries[qid], texts)

            merged = defaultdict(list)
            for cid, sc in zip(chunk_ids, scores):
                merged[cid.split("-")[0].lstrip("doc")].append(sc)  # Basis-ID = Zahl

            ranked = sorted(
                ((doc, sum(vals) / len(vals)) for doc, vals in merged.items()),
                key=lambda x: x[1], reverse=True
            )

            for rank, (doc, score) in enumerate(ranked, 1):
                fout.write(f"{qid} Q0 {doc} {rank} {score:.4f} luyuHF\n")

    print(f"🏁 Finished → {OUT_FILE}")

if __name__ == "__main__":
    main()
