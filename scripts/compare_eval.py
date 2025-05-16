import re
import argparse

def extract_avg_ndcg(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            if "Average nDCG@10" in line:
                match = re.search(r"Average nDCG@10 = ([0-9.]+)", line)
                if match:
                    return float(match.group(1))
    raise ValueError(f"No average nDCG@10 found in {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Compare Lag6 and Lag8 nDCG@10 and compute relative drop")
    parser.add_argument('--lag6', required=True, help='Path to eval file for Lag6')
    parser.add_argument('--lag8', required=True, help='Path to eval file for Lag8')
    parser.add_argument('--output', default='eval_results/eval_bm25_drop.txt', help='Output file for drop result')
    args = parser.parse_args()

    ndcg6 = extract_avg_ndcg(args.lag6)
    ndcg8 = extract_avg_ndcg(args.lag8)

    if ndcg6 == 0:
        drop = float('inf')
    else:
        drop = (ndcg6 - ndcg8) / ndcg6

    result = (
        f"Lag6 Average nDCG@10: {ndcg6:.4f}\n"
        f"Lag8 Average nDCG@10: {ndcg8:.4f}\n"
        f"Relative nDCG@10 Drop: {drop:.2%}\n"
    )

    print(result)
    with open(args.output, 'w') as f:
        f.write(result)

if __name__ == "__main__":
    main()
