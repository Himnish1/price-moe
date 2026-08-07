import re
import csv
import argparse
from pathlib import Path


PATTERN = re.compile(
    r"validation loss at iteration (\d+)\s*\|"
    r"\s*lm loss value:\s*([0-9.E+-]+)\s*\|"
    r"\s*lm loss PPL:\s*([0-9.E+-]+)\s*\|"
)


def parse_log(log_path):
    rows = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                iteration = int(match.group(1))
                lm_loss = float(match.group(2))
                ppl = float(match.group(3))

                rows.append({
                    "iteration": iteration,
                    "lm_loss": lm_loss,
                    "perplexity": ppl,
                })

    return rows


def write_csv(rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iteration", "lm_loss", "perplexity"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output-csv", default="validation_metrics.csv")

    args = parser.parse_args()

    rows = parse_log(args.log_file)

    if not rows:
        print("No validation metrics found.")
        return

    write_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()