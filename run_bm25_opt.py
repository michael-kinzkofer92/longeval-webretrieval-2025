from systems.bm25_baseline.bm25_baseline import BM25Baseline

# === Shared configuration ===
index_path = "data/index"
top_k = 25
k1 = 2.0
b = 0.9

# === Lag6 ===
queries_lag6 = "data/lag6_lag8_subset/French/LongEval Train Collection/Trec/2022-11_fr/queries.trec"
run_file_lag6 = "runs/run_bm25_opt_Lag6.txt"
bm25_lag6 = BM25Baseline(index_path, queries_lag6, run_file_lag6)
bm25_lag6.run_search(k1=k1, b=b, top_k=top_k)

# === Lag8 ===
queries_lag8 = "data/lag6_lag8_subset/French/queries.trec"
run_file_lag8 = "runs/run_bm25_opt_Lag8.txt"
bm25_lag8 = BM25Baseline(index_path, queries_lag8, run_file_lag8)
bm25_lag8.run_search(k1=k1, b=b, top_k=top_k)