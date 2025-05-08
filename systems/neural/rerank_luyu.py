#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyGaggle Luyu‑20‑w06 re‑ranking for LongEval WebRetrieval.

* Reads BM25 run (Top‑25 per query → configurable via TOP_K)
* Loads required TREC documents (release_2025_p1 – June 2022, French)
* Scores with the PyGaggle mono‑BERT model `bert-base-luyu-20w-06`
* Writes re‑ranked run file (TREC format)

Hardware target   : single NVIDIA A40 (48 GB VRAM)
Batch size        : 128 (FP16)
Expected speed    : ≈ 900 query–doc pairs / s  →  75 k queries × 25 docs ≈ 35 min
"""

from pathlib import Path
from typing import Dict, List, Set
import re, json, yaml, torch
from tqdm import tqdm

from pygaggle.rerank.transformer import TransformerReranker
from pygaggle.data.text import Text

# --------------------------------------------------------------------------- #
# Config paths                                                                #
# --------------------------------------------------------------------------- #
CFG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "config.yml"
cfg = yaml.safe_load(CFG_PATH.read_text())

DATA_DIR   = Path(cfg["data"]["data_dir"])
OUTPUT_DIR = Path(cfg["general"]["output_dir"])

DOCUMENT_DIR = Path(
    "data/release_2025_june_subset/release_2025_p1/"
    "French/LongEval Train Collection/Trec/2022-06_fr"
)

BM25_RUN   = OUTPUT_DIR / "run_bm25.txt"
OUTFILE    = OUTPUT_DIR / "run_neural_luyu.txt"

# --------------------------------------------------------------------------- #
# Hyper‑params                                                                #
# --------------------------------------------------------------------------- #
MODEL_NAME   = "pygaggle/bert-base-luyu-20w-06"  # checkpoint used in top CLEF 2024 runs
DEVICE       = "cuda"                            # A40
TOP_K        = 25                                # docs to rerank
BATCH_SIZE   = 128                               # fits A40 in FP16

# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
DOC_START = re.compile(r"<DOC>") ; DOC_END = re.compile(r"</DOC>")
DOCNO     = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.I)

def load_run(path: Path, k: int = TOP_K) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for ln in path.read_text().splitlines():
        qid, _, docid, *_ = ln.split()
        run.setdefault(qid, [])
        if len(run[qid]) < k:
            run[qid].append(docid.strip())
    return run

def parse_queries(path: Path) -> Dict[str, str]:
    mapping, qid = {}, None
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("<num>"):
            qid = (
                ln.replace("<num>","").replace("</num>","")
                  .replace("Number:","").strip()
            )
        elif ln.startswith("<title>"):
            mapping[qid] = ln.replace("<title>","").strip()
    return mapping

def load_docs(directory: Path, needed: Set[str]) -> Dict[str, str]:
    """Return {docid:text} for all IDs in `needed`."""
    plain = {d.lstrip("doc") for d in needed}
    corpus: Dict[str, str] = {}
    for fp in tqdm(directory.rglob("*.trec"), desc="📖 load TREC"):
        with fp.open(encoding="utf-8") as f:
            in_doc, buf, did = False, [], None
            for ln in f:
                if DOC_START.match(ln):
                    in_doc, buf, did = True, [], None ; continue
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
        if len(corpus) >= 2 * len(needed):
            break
    print(f"✅ documents loaded: {len(corpus)//2}")
    return corpus

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    bm25      = load_run(BM25_RUN)
    needed    = {d for lst in bm25.values() for d in lst}
    docs      = load_docs(DOCUMENT_DIR, needed)
    queries   = parse_queries(DATA_DIR / cfg["data"]["queries_file"])

    print("⏳ loading Luyu reranker …")
    reranker = TransformerReranker(
        MODEL_NAME,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        use_fp16=True            # halves VRAM, speeds up 1.7×
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTFILE.open("w") as fout:
        for qid, docids in tqdm(bm25.items(), desc="⚡ rerank"):
            if qid not in queries: continue
            query_text = queries[qid]
            texts = [Text(docs[d], {'docid': d}, 0) for d in docids if d in docs]
            if not texts: continue
            reranked = reranker.rerank(query_text, texts)
            for rank, txt in enumerate(reranked, 1):
                fout.write(
                    f"{qid} Q0 {txt.metadata['docid']} {rank} "
                    f"{txt.score:.4f} luyu20w06\n"
                )
    print(f"🏁 done → {OUTFILE}")

if __name__ == "__main__":
    main()
