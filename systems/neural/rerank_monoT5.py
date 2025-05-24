#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast monoT5 re‑ranking for LongEval WebRetrieval (French, June‑22 subset).

Fast‑path tweaks
----------------
* rerank depth     : 25 docs / query  (TOP_K)
* distilled model  : castorini/monot5-base-msmarco-10k  (~110 M params)
* batch size       : 16, FP16 on CUDA / MPS
* AMP on CUDA      : torch.cuda.amp.autocast() for ~2× speed‑up
* tokenisation uses the fast T5 tokenizer

All paths except DOCUMENT_DIR come from scripts/config.yml.
"""

from pathlib import Path
from typing import Dict, List, Set
import json, re, yaml, torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, T5Tokenizer

# --------------------------------------------------------------------------- #
# Config & constants                                                          #
# --------------------------------------------------------------------------- #
CFG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "config.yml"
cfg = yaml.safe_load(CFG_PATH.read_text())

DATA_DIR   = Path(cfg["data"]["data_dir"])
OUTPUT_DIR = Path(cfg["general"]["output_dir"])

DOCUMENT_DIR = Path(
    "data/release_2025_june_subset/release_2025_p1/"
    "French/LongEval Train Collection/Trec/2022-06_fr"
)

BM25_RUN_FILE  = OUTPUT_DIR / "run_bm25.txt"
OUTPUT_RUNFILE = OUTPUT_DIR / "run_neural_monoT5_2.txt"

MODEL_NAME = "castorini/monot5-base-msmarco-10k"   # distilled 110 M
DEVICE     = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

TOP_K      = 25            # docs per query to rerank
BATCH_SIZE = 64            # fits 16 GB with FP16
AMP        = DEVICE == "cuda"   # autocast works only on CUDA reliably

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_run(path: Path, k: int = TOP_K) -> Dict[str, List[str]]:
    """Read TREC run file -> {qid: [docids]} (max k)."""
    runs: Dict[str, List[str]] = {}
    for line in path.read_text().splitlines():
        qid, _, docid, *_ = line.split()
        runs.setdefault(qid, [])
        if len(runs[qid]) < k:
            runs[qid].append(docid.strip())
    return runs


def parse_queries_trec(trec_path: Path) -> Dict[str, str]:
    """Parse LongEval <num>/<title> file -> {qid: query}."""
    mapping, qid = {}, None
    for line in trec_path.read_text(encoding="utf-8").splitlines():
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


# Quick‑and‑dirty TREC parser -------------------------------------------------
DOC_START = re.compile(r"<DOC>")
DOC_END   = re.compile(r"</DOC>")
DOCNO     = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.I)

def collect_needed_texts(directory: Path, needed: Set[str]) -> Dict[str, str]:
    """
    Scan *.trec files, return {docid/text} for docs in `needed`.
    Adds both 'doc123' and '123' keys so look‑ups always hit.
    """
    corpus, needed_plain = {}, {d.lstrip("doc") for d in needed}
    for fp in tqdm(directory.rglob("*.trec"), desc="📖 Scanning TREC"):
        with fp.open(encoding="utf-8") as f:
            in_doc, buf, doc_id = False, [], None
            for line in f:
                if DOC_START.match(line):
                    in_doc, buf, doc_id = True, [], None
                    continue
                if in_doc and DOC_END.match(line):
                    in_doc = False
                    if doc_id and doc_id.lstrip("doc") in needed_plain:
                        text = " ".join(buf)
                        corpus[doc_id] = text
                        corpus[doc_id.lstrip('doc')] = text
                    continue
                if in_doc:
                    if doc_id is None and (m := DOCNO.search(line)):
                        doc_id = m.group(1).strip()
                    else:
                        buf.append(line.strip())
        if len(corpus) >= 2 * len(needed):
            break
    missing = needed - corpus.keys()
    print(f"✅ Docs loaded: {len(corpus)//2} | missing: {len(missing)}")
    return corpus


def rerank(
    model, tokenizer, query: str, docs: List[str], batch_size: int = BATCH_SIZE
) -> List[float]:
    """Return monoT5 scores for (query, docs)."""
    scores: List[float] = []
    for i in range(0, len(docs), batch_size):
        batch_inputs = [f"Query: {query} Document: {d}" for d in docs[i : i + batch_size]]
        enc = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(DEVICE)


        use_amp = DEVICE in ["cuda", "mps"]

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits
        
            logits = logits[:, 1] if logits.size(-1) > 1 else logits.squeeze(-1)
            scores.extend(logits.cpu().float().tolist())


    return scores

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    bm25_run   = load_run(BM25_RUN_FILE)
    needed_ids = {d for lst in bm25_run.values() for d in lst}
    print(f"🗂️  documents to load: {len(needed_ids):,}")

    corpus  = collect_needed_texts(DOCUMENT_DIR, needed_ids)
    queries = parse_queries_trec(Path("data/lag6_lag8_subset/release_2025_p1/French/queries.trec"))

    print(f"⏳ Loading model {MODEL_NAME} on {DEVICE} …")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        .to(DEVICE)
        .eval()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_RUNFILE.open("w") as fout:
        for qid, docids in tqdm(bm25_run.items(), desc="⚡ Re‑ranking"):
            if qid not in queries:
                continue
            docs_text = [corpus[d] for d in docids if d in corpus]
            if not docs_text:
                continue
            scores = rerank(model, tokenizer, queries[qid], docs_text)
            ranked = sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)
            for rank, (doc, score) in enumerate(ranked, 1):
                fout.write(f"{qid} Q0 {doc} {rank} {score:.4f} monoT5\n")


    print(f"🏁 Finished → {OUTPUT_RUNFILE}")


if __name__ == "__main__":
    main()
