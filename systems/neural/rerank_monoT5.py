#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re‑Ranking der BM25‑Ergebnisse mit monoT5 (RAM‑freundlich).

Pfad‑/Parametersteuerung via scripts/config.yml   – siehe Header deines Originals.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Config laden
# ---------------------------------------------------------------------------
CFG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "config.yml"
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = Path(cfg["data"]["data_dir"])
OUTPUT_DIR = Path(cfg["general"]["output_dir"])
DOCUMENT_DIR = Path(
    "data/release_2025_june_subset/release_2025_p1/French/LongEval Train Collection/Trec/2022-06_fr"
)  # Ordner mit *.json

BM25_RUN_FILE = OUTPUT_DIR / "run_bm25.txt"
OUTPUT_RUN_FILE = OUTPUT_DIR / "run_neural_monoT5.txt"

MODEL_NAME = "castorini/monot5-base-msmarco"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

TOP_K = cfg["bm25"].get("top_k", 100)
BATCH_SIZE = 4  # passt in 16 GB

# ---------------------------------------------------------------------------
# Helfer
# ---------------------------------------------------------------------------
def load_run(path: Path, k: int = TOP_K) -> Dict[str, List[str]]:
    """
    Liest BM25‑Runfile → Dict[qid] = [docids] (Top‑k).
    Entfernt KEINE Teile des Doc‑IDs, merkt sich aber,
    ob ein Präfix 'doc' vorkommt.
    """
    runs: Dict[str, List[str]] = {}
    with path.open() as f:
        for line in f:
            qid, _, docid, _rank, *_ = line.split()
            runs.setdefault(qid, [])
            if len(runs[qid]) < k:
                runs[qid].append(docid.strip())
    return runs


def parse_queries_trec(trec_path: Path) -> Dict[str, str]:
    mapping = {}
    with trec_path.open(encoding="utf-8") as f:
        qid = None
        for line in f:
            if line.startswith("<num>"):
                qid = (
                    line.replace("<num>", "")
                    .replace("</num>", "")
                    .replace("Number:", "")
                    .strip()
                )
            elif line.startswith("<title>"):
                mapping[qid] = line.replace("<title>", "").strip()
    return mapping


import re

DOC_START = re.compile(r"<DOC>")
DOC_END   = re.compile(r"</DOC>")
DOCNO     = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.I)

def collect_needed_texts(directory: Path, needed_ids: set) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    needed_plain = {d.lstrip("doc") for d in needed_ids}

    trec_files = list(directory.rglob("*.trec"))
    for fp in tqdm(trec_files, desc="📖 Scanne TREC‑Dateien"):
        with fp.open(encoding="utf-8") as f:
            in_doc = False
            buf, doc_id = [], None
            for line in f:
                if DOC_START.match(line):
                    in_doc = True
                    buf, doc_id = [], None
                    continue
                if DOC_END.match(line) and in_doc:
                    in_doc = False
                    if doc_id and doc_id.lstrip("doc") in needed_plain:
                        text = " ".join(buf)
                        texts[doc_id] = text
                        texts[doc_id.lstrip('doc')] = text
                    continue
                if in_doc:
                    if doc_id is None:
                        m = DOCNO.search(line)
                        if m:
                            doc_id = m.group(1).strip()
                    else:
                        buf.append(line.strip())

        if len(texts) >= 2 * len(needed_ids):
            break

    missing = needed_ids - texts.keys()
    if missing:
        print(f"⚠️  {len(missing)} Dokumente nicht gefunden.")
    else:
        print("✅ Alle Dokumente gefunden.")
    return texts




def rerank(
    model,
    tokenizer,
    query: str,
    docs: List[str],
    batch_size: int = BATCH_SIZE,
) -> List[float]:
    scores: List[float] = []
    for i in range(0, len(docs), batch_size):
        batch_inputs = [
            f"Query: {query} Document: {d}" for d in docs[i : i + batch_size]
        ]
        enc = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
            logits = logits[:, 1] if logits.size(-1) > 1 else logits.squeeze(-1)
        scores.extend(logits.cpu().tolist())
    return scores


# ---------------------------------------------------------------------------
# Haupt‑Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    # 1) BM25‑Run lesen
    bm25_run = load_run(BM25_RUN_FILE)
    needed_ids = {docid for lst in bm25_run.values() for docid in lst}
    print(f"🗂️  Benötigte Dokumente: {len(needed_ids):,}")

    # 2) Nur diese Dokumenttexte einlesen
    corpus = collect_needed_texts(DOCUMENT_DIR, needed_ids)

    # 3) Queries
    queries = parse_queries_trec(DATA_DIR / cfg["data"]["queries_file"])

    # 4) Modell
    print(f"⏳ Lade monoT5 auf {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE in {"mps", "cuda"} else None,
    ).to(DEVICE).eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5) Re‑Ranking
    with OUTPUT_RUN_FILE.open("w", encoding="utf-8") as fout:
        for qid, docids in tqdm(bm25_run.items(), desc="Re‑Ranking"):
            if qid not in queries:
                continue
            docs_text = [corpus[d] for d in docids if d in corpus]
            if not docs_text:
                continue
            scores = rerank(model, tokenizer, queries[qid], docs_text)
            ranked = sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, 1):
                fout.write(f"{qid} Q0 {docid} {rank} {score:.4f} monoT5\n")

    print(f"✅ Re‑Ranking fertig → {OUTPUT_RUN_FILE}")


if __name__ == "__main__":
    main()
