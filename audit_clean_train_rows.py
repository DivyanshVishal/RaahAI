"""Count station-stop rows per train in the cleaned 2026 Parquet dataset."""

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent


def count_partition(path):
    counts = Counter()
    with pq.ParquetFile(path) as source:
        if "train_no" not in source.schema_arrow.names:
            raise ValueError(f"Missing train_no column in {path}")
        for batch in source.iter_batches(columns=["train_no"], use_threads=False):
            numbers = pc.utf8_trim_whitespace(batch.column(0))
            for item in pc.value_counts(numbers).to_pylist():
                if not item["values"]:
                    raise ValueError(f"Missing train_no in {path}")
                counts[item["values"]] += item["counts"]
        if sum(counts.values()) != source.metadata.num_rows:
            raise ValueError(f"Row count differs from Parquet metadata in {path}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "processed/clean_data/year=2026")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_audit/clean_2026_train_row_counts.csv")
    args = parser.parse_args()
    files = sorted(p for p in args.input_dir.rglob("*.parquet") if p.name != "_stations.parquet")
    if not files:
        parser.error(f"No train partitions found under {args.input_dir}")
    if args.output.suffix.lower() != ".csv" or args.output.resolve().is_relative_to(args.input_dir.resolve()):
        parser.error("Choose a CSV output file outside the input dataset")

    counts = Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for partition in pool.map(count_partition, files):
            counts.update(partition)

    ranking = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=args.output.parent, delete=False)
    try:
        with temp:
            writer = csv.writer(temp)
            writer.writerow(["train_no", "row_count"])
            writer.writerows(ranking)
        Path(temp.name).replace(args.output)
    finally:
        Path(temp.name).unlink(missing_ok=True)
    print(f"{len(files):,} partitions | {len(counts):,} trains | {sum(counts.values()):,} stop rows")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
