#!/usr/bin/env bash
set -e

declare -A MODELS EMB FAISS PCA
MODELS[01]="distiluse-base-multilingual-cased-v1"
EMB[01]=cls;     FAISS[01]=flat; PCA[01]=
MODELS[02]="distiluse-base-multilingual-cased-v1"
EMB[02]=mean;    FAISS[02]=flat; PCA[02]=
MODELS[03]="paraphrase-multilingual-MiniLM-L12-v2"
EMB[03]=mean;    FAISS[03]=flat; PCA[03]=
MODELS[04]="flax-community/camembert-base-sentence"
EMB[04]=cls;     FAISS[04]=flat; PCA[04]=
MODELS[05]="distiluse-base-multilingual-cased-v1"
EMB[05]=mean;    FAISS[05]=hnsw; PCA[05]=
MODELS[06]="flax-community/camembert-base-sentence"
EMB[06]=mean;    FAISS[06]=hnsw; PCA[06]=
MODELS[07]="distiluse-base-multilingual-cased-v1"
EMB[07]=mean;    FAISS[07]=flat; PCA[07]= 
MODELS[08]="distiluse-base-multilingual-cased-v1"
EMB[08]=mean;    FAISS[08]=flat; PCA[08]=128

for ID in 08; do
  echo -e "\\n▶▶ Lauf $ID  – ${MODELS[$ID]} (${EMB[$ID]}, ${FAISS[$ID]}, PCA=${PCA[$ID]:-none})"
  python systems/representation/run_dense_pipeline.py \
    --run-id "$ID" \
    --model "${MODELS[$ID]}" \
    --embedding "${EMB[$ID]}" \
    --faiss "${FAISS[$ID]}" \
    ${PCA[$ID]:+--dim-reduce "${PCA[$ID]}"} \
    --topk 10 \
    --evaluate
done
