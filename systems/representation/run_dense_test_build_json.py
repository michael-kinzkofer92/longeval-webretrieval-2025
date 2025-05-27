#!/usr/bin/env python3
"""
Dense Retrieval für LongEval-TEST (TREC- oder JSON-Fassung).

Erweiterungen ggü. run_dense_test_build_debug.py:
• erkennt Sammlungen auch unter  .../Json/<month>_fr/collection/
• liest .jsonl, .jsonl.gz, .json, .jsonl.json
  – egal ob als echtes JSONL ODER als ein einziges JSON-Array.
• viele Debug-Ausgaben via --verbose.
"""
from __future__ import annotations
import argparse, json, gzip, io, os, pickle, re
from pathlib import Path
from typing import Dict, List

import numpy as np
import faiss, torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
faiss.omp_set_num_threads(os.cpu_count())
ROOT = Path(__file__).resolve().parents[2]

# ----------------------------- Query-Loader -------------------------------- #
def read_queries_txt(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return {qid: txt for qid, txt in
                (ln.rstrip("\n").split("\t", 1) for ln in f if ln.strip())}

# ------------------------------ I/O Helfer --------------------------------- #
DOC_START_RE = re.compile(r"\s*<DOC>",  re.I)
DOC_END_RE   = re.compile(r"\s*</DOC>", re.I)
DOCNO_RE     = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>", re.I)

def smart_open(path: Path) -> io.TextIOBase:
    """Öffnet Text- oder Gzip-Datei transparent."""
    with path.open("rb") as fh:
        return (gzip.open if fh.read(2) == b"\x1f\x8b" else open)(
            path, "rt", encoding="utf-8", errors="ignore")

def _add_doc(corpus: Dict[str, str], obj):
    """Robust: akzeptiert dict ODER plain string."""
    # -------- Fall 1: plain String ---------------------------------------
    if isinstance(obj, str):
        if obj.strip():
            did = f"auto_{len(corpus)}"
            corpus[did] = obj.strip()
        return

    # -------- Fall 2: wir brauchen ein Dict ------------------------------
    if not isinstance(obj, dict):
        return                              # Listen, Zahlen, None → ignorieren

    # verschachtelte Metadaten einebnen
    for k in ("document", "data", "metadata"):
        if isinstance(obj.get(k), dict):
            obj = {**obj, **obj[k]}

    did = (obj.get("docno")  or obj.get("id") or obj.get("_id") or
           obj.get("doc_id") or obj.get("docId") or obj.get("url"))
    txt = (obj.get("contents") or obj.get("text") or obj.get("body") or
           obj.get("content")  or obj.get("article") or obj.get("body_text") or
           obj.get("plain_text"))

    if txt and not did:
        did = f"auto_{len(corpus)}"
    if did and txt:
        corpus[did] = txt



def parse_trec_filelike(file_obj, verbose=False) -> Dict[str, str]:
    corpus = {}; in_doc = False; buf = []; did = None
    for ln in file_obj:
        if ln.startswith("<DOC>"):
            in_doc, buf, did = True, [], None;  continue
        if ln.startswith("</DOC>"):
            in_doc = False
            if did: corpus[did] = " ".join(buf)
            continue
        if in_doc:
            if ln.startswith("<DOCNO>"):
                did = ln.replace("<DOCNO>","").replace("</DOCNO>","").strip()
            elif not ln.startswith("<"):
                buf.append(ln.strip())
    if verbose:
        print(f"    ↪︎ {len(corpus):,} TREC-Docs extrahiert")
    return corpus


# ---------------------------- Haupt-Parser --------------------------------- #
# ---------------------------- Haupt-Parser --------------------------------- #
def parse_docs_anyformat(collection_dir: Path, verbose=False) -> Dict[str, str]:
    corpus: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # 1) Alle Kandidat-Dateien zusammenstellen (json*, trec-Verzeichnis) #
    # ------------------------------------------------------------------ #
    pattern = ["*.jsonl*", "*.json", "*.jsonl", "*.jsonl.gz"]
    json_files: List[Path] = []
    for pat in pattern:
        json_files += collection_dir.glob(pat)
        json_files += (collection_dir / "collection").glob(pat)

    trec_files: List[Path] = list(collection_dir.rglob("*.trec"))

    # -------------- 2) Erst echte *.trec normal einlesen --------------- #
    for fp in trec_files:
        if verbose:
            print("🔍 TREC", fp.relative_to(collection_dir.parent.parent))
        with fp.open(encoding="utf-8") as fh:
            corpus.update(parse_trec_filelike(fh, verbose))

    # ---- 3) Dann alle json*/jsonl* - Dateien inhaltsbasiert parsen ---- #
    for fp in json_files:
        if verbose:
            print("🔍 Datei", fp.relative_to(collection_dir.parent.parent))
        try:
            with smart_open(fp) as f:
                sample = f.read(1000)          # erste 1000 B ansehen
                f.seek(0)

                first = sample.lstrip()[:1]
                if first == '<':               # → TREC trotz json-Endung
                    if verbose:
                        print("    ↪︎ erkannt als TREC (falsche Endung)")
                    corpus.update(parse_trec_filelike(f, verbose))
                    continue

                if first == '[':               # → einziges JSON-Array
                    try:
                        for obj in json.load(f):
                            _add_doc(corpus, obj)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print("⚠️  JSON-Array defekt:", e)
                    continue

                # ----------- echtes JSONL (eine Zeile = ein Objekt) --------
                for ln_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        _add_doc(corpus, json.loads(line))
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"⚠️  Zeile {ln_no} ignoriert – kein JSON")

        except Exception as e:
            if verbose:
                print(f"⚠️  Fehler beim Lesen {fp.name}: {e}")

    return corpus



    # ---------- JSON / JSONL / Arrays ------------------------------------- #
    pattern = ["*.jsonl*", "*.json"]          # *.jsonl, *.jsonl.gz, *.jsonl.json, *.json
    json_files: List[Path] = []
    for pat in pattern:
        json_files += collection_dir.glob(pat)
        json_files += (collection_dir / "collection").glob(pat)

    for fp in json_files:
        if verbose: print("🔍 JSON ", fp.relative_to(collection_dir.parent.parent))
        try:
            with smart_open(fp) as f:
                first = f.read(1)
                f.seek(0)
                if first.lstrip().startswith('['):
                    # ------- 1 grosses Array --------------------------------
                    try:
                        objs = json.load(f)
                        for o in objs:
                            _add_doc(corpus, o)
                    except json.JSONDecodeError as e:
                        if verbose: print(f"⚠️  JSON-Array defekt: {e}")
                    continue

                # -------- echtes JSONL --------------------------------------
                for ln_no, line in enumerate(f, 1):
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        _add_doc(corpus, obj)
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"⚠️  (ignoriert) Zeile {ln_no} nicht valide JSONL")
        except Exception as e:
            if verbose: print(f"⚠️  Fehler beim Lesen {fp.name}: {e}")

    return corpus

# ----------------------------- Embedding etc. ------------------------------ #
def embed(model, texts, batch):
    vecs = []
    for i in tqdm(range(0, len(texts), batch), desc="Embedding", unit="batch"):
        vecs.append(model.encode(texts[i:i+batch],
                                 convert_to_numpy=True, show_progress_bar=False))
    return np.vstack(vecs)

def ensure_faiss_index(month, texts, docids, model, batch, dim_reduce, verbose=False):
    idx_dir = ROOT / f"data/{month}_index";  idx_dir.mkdir(parents=True, exist_ok=True)
    f_index, f_docs, f_ids, f_pca = [idx_dir / n for n in
        ("index.faiss", "doc_embeddings.npy", "docids.npy", "pca.pkl")]

    if f_docs.exists() and f_ids.exists():
        if verbose: print("📂 Lade gecachte Embeddings")
        doc_emb = np.load(f_docs);  np_ids = np.load(f_ids)
        assert len(np_ids) == len(docids)
    else:
        print("🧮  Embedding Dokumente …")
        doc_emb = embed(model, texts, batch)
        np.save(f_docs, doc_emb); np.save(f_ids, np.array(docids))

    dim = doc_emb.shape[1]
    if dim_reduce and dim_reduce < dim:

        if not f_pca.exists():
            print(f"🔻  PCA → {dim_reduce} Dim.")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=dim_reduce, random_state=42)
            doc_emb = pca.fit_transform(doc_emb)
            with open(f_pca, "wb") as f: pickle.dump(pca, f)
        else:
            if verbose: print("⚙️  Lade PCA")
            with open(f_pca, "rb") as f: pca = pickle.load(f)
            doc_emb = pca.transform(doc_emb).astype("float32")
    
        dim = dim_reduce

    
    if f_index.exists():
        if verbose: print("📂 Lade FAISS-Index")
        index = faiss.read_index(str(f_index))
    else:
        print("🔧  Baue FAISS-Index …")
        index = faiss.IndexFlatIP(dim)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(doc_emb.astype("float32"));  faiss.write_index(index, str(f_index))
    return index, doc_emb

# ------------------------------ Monatslauf --------------------------------- #
def find_collection_dir(month_hy: str) -> Path | None:
    """Suche sowohl Trec/… als auch Json/…"""
    cand = [
        ROOT / f"data/test/LongEval Test Collection/Trec/{month_hy}_fr",
        ROOT / f"data/test/LongEval Test Collection/Json/{month_hy}_fr",
    ]
    for c in cand:
        if c.exists(): return c
    return None

def process_month(month: str, args, model):
    month_hy = month.replace("_", "-")
    print(f"\n================ {month_hy} ================")

    col_dir = find_collection_dir(month_hy)
    if not col_dir:
        print("✖  Weder Trec- noch Json-Verzeichnis gefunden – übersprungen.")
        return
    if args.verbose:
        print("📂  benutze Sammlung:", col_dir.relative_to(ROOT))

    query_fp = ROOT / f"data/test/LongEval Test Collection/queries/{month_hy}_queries.txt"
    if not query_fp.exists():
        print(f"✖  Query-Datei {query_fp} fehlt – übersprungen.")
        return

    # ---------------- Dokumente ------------------------------------------- #
    print("📑  Lese Dokumente …")
    corpus = parse_docs_anyformat(col_dir, verbose=args.verbose)
    if not corpus:
        print("⚠️  Kein Dokument eingelesen – bitte --verbose prüfen.")
        return
    if args.verbose:
        print(f"✅  {len(corpus):,} Dokumente geladen; Beispiel:")
        for i, (k, v) in enumerate(corpus.items()):
            print(f"    {k[:25]:<25} | {v[:70].replace(chr(10),' ')}")
            if i == 2: break

    docids, texts = list(corpus.keys()), list(corpus.values())
    index, _ = ensure_faiss_index(month, texts, docids, model,
                                  args.batch_size, args.dim_reduce, args.verbose)

    # ---------------- Queries --------------------------------------------- #
    queries = read_queries_txt(query_fp)
    qids, qtexts = zip(*queries.items())
    print(f"▶  {len(qids)} Queries   |   {index.ntotal:,} Doc-Vektoren")

    q_emb = embed(model, list(qtexts), args.batch_size)
    if args.dim_reduce:
        pca_file = ROOT / f"data/{month}_index/pca.pkl"
        if pca_file.exists():
            with open(pca_file, "rb") as f: pca = pickle.load(f)
            q_emb = pca.transform(q_emb).astype("float32")

    sim, idxs = index.search(q_emb.astype("float32"), args.topk)

    out_dir = ROOT / f"web-submission-dense/{month_hy}"; out_dir.mkdir(parents=True, exist_ok=True)
    run_fp  = out_dir / f"run_dense_{args.run_id}.txt"
    with run_fp.open("w") as fout:
        for qi, qid in enumerate(qids):
            for rank, (didx, score) in enumerate(zip(idxs[qi], sim[qi]), 1):
                fout.write(f"{qid} Q0 {docids[didx]} {rank} {score:.4f} dense_{args.run_id}\n")
    print("✅  Run geschrieben →", run_fp)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--embedding", choices=["cls", "mean"], default="cls")
    ap.add_argument("--dim-reduce", type=int)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--months", nargs="+",
                    default=["2023_03","2023_04","2023_05",
                             "2023_06","2023_07","2023_08"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔤  Lade SBERT {args.model} auf {device}")
    sbert = SentenceTransformer(args.model, device=device)

    for m in args.months:
        process_month(m, args, sbert)
