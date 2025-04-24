import argparse
import pytrec_eval

def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

def load_run(run_file):
    run = {}
    with open(run_file, 'r') as f:
        for line in f:
            qid, _, docid, _, score, _ = line.strip().split()
            run.setdefault(qid, {})[docid] = float(score)
    return run

def evaluate(qrels_file, run_file):
    qrels_data = load_qrels(qrels_file)
    run_data = load_run(run_file)

    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_data, {'ndcg_cut.10'})
    results = evaluator.evaluate(run_data)

    # Output per query
    for qid, metric_values in results.items():
        print(f'{qid}: nDCG@10 = {metric_values["ndcg_cut_10"]:.4f}')

    # Calculate average
    avg_ndcg = sum(m["ndcg_cut_10"] for m in results.values()) / len(results)
    print(f'\nAverage nDCG@10 = {avg_ndcg:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate run file against qrels using nDCG@10")
    parser.add_argument("--qrels", required=True, help="Pfad zur Qrels-Datei (TREC-Format)")
    parser.add_argument("--run", required=True, help="Pfad zur Run-Datei (TREC-Format)")

    args = parser.parse_args()
    evaluate(args.qrels, args.run)
