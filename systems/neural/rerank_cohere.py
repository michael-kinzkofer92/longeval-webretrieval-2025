#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cohere Rerank (up to 100 docs) for LongEval WebRetrieval ‑ French, June‑22.

* liest runs/run_bm25.txt   (Top‑N pro Query, N = TOP_K)
* lädt nur die wirklich benötigten Dokument‑Texte aus den .trec‑Files
* ruft Cohere‑/rerank‑API auf  (model="rerank-multilingual-v3.0")
* schreibt runs/run_neural_cohere.txt im TREC‑Format
"""
from pathlib import Path
from typing import Dict, List, Set
import re, os, yaml, cohere, tqdm, textwrap

# --------------------------------------------------------------------------- #
# Konfigpfade                                                                 #
# --------------------------------------------------------------------------- #
CFG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "config.yml"
cfg      = yaml.safe_load(CFG_PATH.read_text())

DATA_DIR   = Path(cfg["data"]["data_dir"])
OUTPUT_DIR = Path(cfg["general"]["output_dir"])

DOCUMENT_DIR = Path(
    "data/release_2025_june_subset/release_2025_p1/"
    "French/LongEval Train Collection/Trec/2022-06_fr"
)

BM25_RUN   = OUTPUT_DIR / "run_bm25.txt"
OUT_FILE   = OUTPUT_DIR / "run_neural_cohere.txt"

COHERE_MODEL = "rerank-multilingual-v3.0"
TOP_K        = 25                     # docs per query to rerank (kommt aus BM25‑Run)
BATCH_SIZE   = 100                    # Cohere akzeptiert bis 100 Paarungen pro Call

# --------------------------------------------------------------------------- #
# kleine Helfer                                                               #
# --------------------------------------------------------------------------- #
DOC_START = re.compile(r"<DOC>");  DOC_END = re.compile(r"</DOC>")
DOCNO     = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.I)

def load_bm25(path: Path, k: int = TOP_K) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for ln in path.read_text().splitlines():
        qid, _, docid, *_ = ln.split()
        run.setdefault(qid, []).append(docid)
    # nur Top‑k behalten
    return {q: docs[:k] for q, docs in run.items()}

def parse_queries(trec: Path) -> Dict[str, str]:
    mapping, qid = {}, None
    for ln in trec.read_text(encoding="utf-8").splitlines():
        if ln.startswith("<num>"):
            qid = ln.replace("<num>","").replace("</num>","").replace("Number:","").strip()
        elif ln.startswith("<title>"):
            mapping[qid] = ln.replace("<title>","").strip()
    return mapping

def load_docs(directory: Path, needed: Set[str]) -> Dict[str, str]:
    plain = {d.lstrip("doc") for d in needed}
    corpus: Dict[str, str] = {}
    for fp in tqdm.tqdm(list(directory.rglob("*.trec")), desc="📖 reading .trec"):
        with fp.open(encoding="utf-8") as f:
            in_doc, buf, did = False, [], None
            for ln in f:
                if DOC_START.match(ln):
                    in_doc, buf, did = True, [], None;  continue
                if in_doc and DOC_END.match(ln):
                    in_doc = False
                    if did and did.lstrip("doc") in plain:
                        corpus[did] = " ".join(buf)
                        corpus[did.lstrip("doc")] = corpus[did]   # dup‑key ohne "doc"
                    continue
                if in_doc:
                    if did is None and (m := DOCNO.search(ln)):
                        did = m.group(1).strip()
                    else:
                        buf.append(ln.strip())
        if len(corpus) >= 2 * len(needed):
            break
    print(f"✅ loaded {len(corpus)//2} documents")
    return corpus

# --------------------------------------------------------------------------- #
# Hauptlogik                                                                  #
# --------------------------------------------------------------------------- #
def main() -> None:
    # --- Dateien laden ----------------------------------------------------- #
    bm25    = load_bm25(BM25_RUN)
    needed  = {d for lst in bm25.values() for d in lst}
    docs    = load_docs(DOCUMENT_DIR, needed)
    queries = parse_queries(DATA_DIR / cfg["data"]["queries_file"])

    # --- Cohere‑Client ----------------------------------------------------- #
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Bitte COHERE_API_KEY als Umgebungsvariable setzen!")
    coh = cohere.Client(api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as fout:
        for qid, docids in tqdm.tqdm(bm25.items(), desc="⚡ Cohere rerank"):
            if qid not in queries: continue
            texts = [docs[d] for d in docids if d in docs]
            if not texts: continue

            # --- API‑Call --------------------------------------------------- #
            resp = coh.rerank(
                query       = queries[qid],
                documents   = texts,
                top_n       = len(texts),          # vollständige Sortierung
                model       = COHERE_MODEL,
                return_documents = False
            )
            # resp.results enthält eine Liste mit index + relevance_score

            ranked = [(docids[r.index], r.relevance_score) for r in resp.results]

            for rank, (docid, score) in enumerate(ranked, 1):
                fout.write(f"{qid} Q0 {docid.lstrip('doc')} {rank} {score:.4f} cohere\n")

    print(f"🏁 Finished → {OUT_FILE}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
