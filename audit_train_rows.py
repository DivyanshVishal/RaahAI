"""Count all stop records by train number in the raw Indian Railways CSV dataset."""

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as arrow_csv


ROOT = Path(__file__).resolve().parent


def count_file(path):
    counts = Counter()
    try:
        with arrow_csv.open_csv(
            path,
            parse_options=arrow_csv.ParseOptions(newlines_in_values=True),
            convert_options=arrow_csv.ConvertOptions(
                include_columns=["train_no"], column_types={"train_no": pa.string()}
            ),
        ) as reader:
            for batch in reader:
                numbers = pc.utf8_trim_whitespace(batch.column(0))
                for entry in pc.value_counts(numbers).to_pylist():
                    if not entry["values"]:
                        raise ValueError("missing train_no")
                    counts[entry["values"]] += entry["counts"]
    except (ValueError, OSError, KeyError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "ETADataset/trains")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_audit/train_row_counts.csv")
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("*.csv"))
    if not files:
        parser.error(f"No CSV files in {args.input_dir}")
    if args.output.resolve().parent == args.input_dir.resolve():
        parser.error("Choose an output path outside the input directory")

    counts = Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for file_counts in pool.map(count_file, files):
            counts.update(file_counts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                  dir=args.output.parent, delete=False)
    temp_path = Path(temporary.name)
    try:
        with temporary:
            writer = csv.writer(temporary)
            writer.writerow(["train_no", "row_count"])
            writer.writerows(sorted(counts.items()))
        temp_path.replace(args.output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"{len(files):,} files | {len(counts):,} trains | {sum(counts.values()):,} rows")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
