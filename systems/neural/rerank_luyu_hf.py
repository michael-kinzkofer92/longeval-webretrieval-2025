#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light‑weight HuggingFace version of the ‘Luyu‑20‑w06’ mono‑BERT reranker
for LongEval WebRetrieval – **no PyGaggle required**.

Input  : runs/run_bm25.txt   (Top‑25 docs / query)
Output : runs/run_neural_luyu.txt  (TREC‑format)

≈ 35 min on a single NVIDIA A40 (48 GB, FP16, batch 128)
≈ 6–7 h on 8‑core CPU (batch 8, FP32)
"""

from pathlib import Path
from typing import Dict, List, Set
import re, yaml, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------------------- #
# Config                                                                    #
# ------------------------------------------------------------------------- #
CFG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "config.yml"
cfg      = yaml.safe_load(CFG_PATH.read_text())

DATA_DIR   = Path(cfg["data"]["data_dir"])
OUTPUT_DIR = Path(cfg["general"]["output_dir"])

DOCUMENT_DIR = Path(
    "data/release_2025_june_subset/release_2025_p1/"
    "French/LongEval Train Collection/Trec/2022-06_fr"
)

BM25_RUN   = OUTPUT_DIR / "run_bm25.txt"
OUT_FILE   = OUTPUT_DIR / "run_neural_luyu.txt"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # frei verfügbar
TOP_K      = 25
BATCH_SIZE = 128                             # adjust downwards for CPU / small GPUs

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
USE_FP16 = DEVICE == "cuda"                  # AMP only safe on CUDA

# ------------------------------------------------------------------------- #
# Helpers                                                                   #
# ------------------------------------------------------------------------- #
DOC_START = re.compile(r"<DOC>");  DOC_END = re.compile(r"</DOC>")
DOCNO     = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.I)

def load_run(path: Path, k: int = TOP_K) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for ln in path.read_text().splitlines():
        qid, _, docid, *_ = ln.split()
        run.setdefault(qid, [])
        if len(run[qid]) < k:
            run[qid].append(docid.strip())
    return run

def parse_queries(trec: Path) -> Dict[str, str]:
    mapping, qid = {}, None
    for ln in trec.read_text(encoding="utf-8").splitlines():
        if ln.startswith("<num>"):
            qid = ln.replace("<num>","").replace("</num>","") \
                   .replace("Number:","").strip()
        elif ln.startswith("<title>"):
            mapping[qid] = ln.replace("<title>","").strip()
    return mapping

def load_docs(directory: Path, needed: Set[str]) -> Dict[str, str]:
    plain = {d.lstrip("doc") for d in needed}
    corpus: Dict[str,str] = {}
    for fp in tqdm(directory.rglob("*.trec"), desc="📖 scanning TREC"):
        with fp.open(encoding="utf-8") as f:
            in_doc, buf, did = False, [], None
            for ln in f:
                if DOC_START.match(ln):
                    in_doc, buf, did = True, [], None;  continue
                if in_doc and DOC_END.match(ln):
                    in_doc = False
                    if did and did.lstrip("doc") in plain:
                        text = " ".join(buf)
                        corpus[did] = text
                        corpus[did.lstrip("doc")] = text
                    continue
                if in_doc:
                    if did is None and (m:=DOCNO.search(ln)):
                        did = m.group(1).strip()
                    else:
                        buf.append(ln.strip())
        if len(corpus) >= 2*len(needed):
            break
    print(f"✅ loaded {len(corpus)//2} documents")
    return corpus

def rerank(model, tok, query: str, docs: List[str]) -> List[float]:
    scores: List[float] = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        enc = tok(
            [f"Query: {query} Document: {d}" for d in batch],
            padding = True,
            truncation = True,
            max_length = 256,
            return_tensors = "pt"
        ).to(DEVICE)

        with torch.no_grad():
            if USE_FP16:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits
        # binary classifier: positive class is index 1
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
    queries= parse_queries(DATA_DIR / cfg["data"]["queries_file"])

    print(f"⏳ loading {MODEL_NAME} on {DEVICE} …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if USE_FP16 else None)
        .to(DEVICE)
        .eval()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as fout:
        for qid, docids in tqdm(bm25.items(), desc="⚡ reranking"):
            if qid not in queries: continue
            texts = [docs[d] for d in docids if d in docs]
            if not texts: continue
            scores = rerank(model, tok, queries[qid], texts)
            ranked = sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, 1):
                fout.write(f"{qid} Q0 {docid.lstrip('doc')} {rank} {score:.4f} luyuHF\n")

    print(f"🏁 Finished → {OUT_FILE}")

if __name__ == "__main__":
    main()
