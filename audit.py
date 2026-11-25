from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import json
import hashlib
import time
import csv


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS
DATASET_DIR = Path(r"C:\System\ProjectV\ETADataset\trains")

# Output directory
OUTPUT_DIR = Path("dataset_audit")
OUTPUT_DIR.mkdir(exist_ok=True)

# File extensions to scan
EXTENSIONS = {".csv"}

# Process CSV in chunks so large files don't consume RAM
CHUNK_SIZE = 50_000


# ============================================================
# EXPECTED SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "date",
    "train_no",
    "train_name",
    "train_type",
    "seq",
    "station_code",
    "halts_here",
    "sch_arrival",
    "sch_arrival_date",
    "sch_departure",
    "sch_departure_date",
    "act_arrival",
    "act_arrival_date",
    "act_departure",
    "act_departure_date",
    "delay_arrival_min",
    "delay_departure_min",
    "platform",
    "distance_km",
    "lat",
    "lng",
    "departed",
    "reported",
    "reported_at",
]


# ============================================================
# HELPERS
# ============================================================

def clean_column_name(col):
    """
    Clean malformed column names that may come from
    copied markdown/HTML.
    """
    col = str(col).strip()

    # Remove markdown bold markers
    col = col.replace("**", "")

    # Common HTML entity corruption
    replacements = {
        "&#x73;": "s",
        "&#x31;": "1",
        "&#x6C;": "l",
        "&#x6c;": "l",
    }

    for old, new in replacements.items():
        col = col.replace(old, new)

    return col.strip()


def normalize_columns(columns):
    return [clean_column_name(c) for c in columns]


def safe_read_header(filepath):
    """
    Read only the header.
    """
    try:
        df = pd.read_csv(filepath, nrows=0)
        df.columns = normalize_columns(df.columns)
        return list(df.columns), None
    except Exception as e:
        return None, str(e)


def parse_date_series(series):
    """
    Parse dates safely.
    """
    return pd.to_datetime(series, errors="coerce")


def file_hash(filepath, chunk_size=1024 * 1024):
    """
    MD5 hash of a file.
    Used only for detecting exact duplicate files.
    """
    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("RAAHAI DATASET AUDIT")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"\nERROR: Dataset directory does not exist:")
        print(DATASET_DIR)
        return

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------

    files = [
        p for p in DATASET_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    print(f"\nDataset directory : {DATASET_DIR}")
    print(f"CSV files found   : {len(files)}")

    if not files:
        print("\nNo CSV files found.")
        return

    # --------------------------------------------------------
    # GLOBAL COUNTERS
    # --------------------------------------------------------

    total_rows = 0
    total_size_bytes = 0

    unique_trains = set()
    unique_stations = set()
    unique_journeys = set()

    date_min = None
    date_max = None

    actual_arrival_count = 0
    actual_departure_count = 0
    gps_count = 0
    reported_at_count = 0

    missing_counts = Counter()

    schema_counter = Counter()
    bad_files = []

    file_statistics = []

    train_counter = Counter()
    station_counter = Counter()

    train_rows = Counter()
    station_rows = Counter()

    duplicate_row_count = 0

    # We don't store all rows in RAM.
    # Instead we hash important row combinations.
    # This detects repeated records across files/chunks.
    row_hash_counts = Counter()

    # Repeated train + station observations
    train_station_counts = Counter()

    # Exact duplicate file hashes
    file_hashes = defaultdict(list)

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    for file_index, filepath in enumerate(files, start=1):

        relative_path = filepath.relative_to(DATASET_DIR)

        try:
            size_bytes = filepath.stat().st_size
        except Exception:
            size_bytes = 0

        total_size_bytes += size_bytes

        print(
            f"\rProcessing {file_index}/{len(files)} "
            f"| rows={total_rows:,}",
            end="",
            flush=True
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        columns, header_error = safe_read_header(filepath)

        if header_error:
            bad_files.append({
                "file": str(relative_path),
                "reason": f"Header error: {header_error}"
            })
            continue

        schema_tuple = tuple(columns)
        schema_counter[schema_tuple] += 1

        # ----------------------------------------------------
        # SCHEMA CHECK
        # ----------------------------------------------------

        missing_columns = [
            c for c in EXPECTED_COLUMNS
            if c not in columns
        ]

        extra_columns = [
            c for c in columns
            if c not in EXPECTED_COLUMNS
        ]

        schema_ok = (
            not missing_columns
            and not extra_columns
        )

        if not schema_ok:

            bad_files.append({
                "file": str(relative_path),
                "reason": "Schema mismatch",
                "missing_columns": ",".join(missing_columns),
                "extra_columns": ",".join(extra_columns),
            })

        # ----------------------------------------------------
        # READ IN CHUNKS
        # ----------------------------------------------------

        file_rows = 0

        file_actual_arrival = 0
        file_actual_departure = 0
        file_gps = 0
        file_reported_at = 0

        file_dates = []

        try:

            for chunk in pd.read_csv(
                filepath,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ):

                # Normalize columns
                chunk.columns = normalize_columns(chunk.columns)

                rows = len(chunk)
                file_rows += rows
                total_rows += rows

                # ------------------------------------------------
                # DATE
                # ------------------------------------------------

                if "date" in chunk.columns:

                    dates = parse_date_series(chunk["date"])

                    valid_dates = dates.dropna()

                    if len(valid_dates) > 0:

                        local_min = valid_dates.min()
                        local_max = valid_dates.max()

                        file_dates.append(
                            (local_min, local_max)
                        )

                        if date_min is None or local_min < date_min:
                            date_min = local_min

                        if date_max is None or local_max > date_max:
                            date_max = local_max

                # ------------------------------------------------
                # UNIQUE TRAINS
                # ------------------------------------------------

                if "train_no" in chunk.columns:

                    vals = (
                        chunk["train_no"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    unique_trains.update(vals.unique())

                    train_counter.update(vals)

                # ------------------------------------------------
                # UNIQUE STATIONS
                # ------------------------------------------------

                if "station_code" in chunk.columns:

                    vals = (
                        chunk["station_code"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    unique_stations.update(vals.unique())

                    station_counter.update(vals)

                # ------------------------------------------------
                # UNIQUE JOURNEYS
                # ------------------------------------------------

                if (
                    "train_no" in chunk.columns
                    and "date" in chunk.columns
                ):

                    journey_values = (
                        chunk["train_no"]
                        .astype(str)
                        .str.strip()
                        + "|"
                        + chunk["date"]
                        .astype(str)
                        .str.strip()
                    )

                    unique_journeys.update(
                        journey_values.dropna().unique()
                    )

                # ------------------------------------------------
                # ACTUAL ARRIVAL
                # ------------------------------------------------

                if "act_arrival" in chunk.columns:

                    valid = (
                        chunk["act_arrival"]
                        .notna()
                        & (
                            chunk["act_arrival"]
                            .astype(str)
                            .str.strip()
                            != ""
                        )
                    )

                    count = int(valid.sum())

                    actual_arrival_count += count
                    file_actual_arrival += count

                # ------------------------------------------------
                # ACTUAL DEPARTURE
                # ------------------------------------------------

                if "act_departure" in chunk.columns:

                    valid = (
                        chunk["act_departure"]
                        .notna()
                        & (
                            chunk["act_departure"]
                            .astype(str)
                            .str.strip()
                            != ""
                        )
                    )

                    count = int(valid.sum())

                    actual_departure_count += count
                    file_actual_departure += count

                # ------------------------------------------------
                # GPS
                # ------------------------------------------------

                if (
                    "lat" in chunk.columns
                    and "lng" in chunk.columns
                ):

                    valid = (
                        pd.to_numeric(
                            chunk["lat"],
                            errors="coerce"
                        ).notna()
                        &
                        pd.to_numeric(
                            chunk["lng"],
                            errors="coerce"
                        ).notna()
                    )

                    count = int(valid.sum())

                    gps_count += count
                    file_gps += count

                # ------------------------------------------------
                # REPORTED AT
                # ------------------------------------------------

                if "reported_at" in chunk.columns:

                    valid = (
                        chunk["reported_at"]
                        .notna()
                        & (
                            chunk["reported_at"]
                            .astype(str)
                            .str.strip()
                            != ""
                        )
                    )

                    count = int(valid.sum())

                    reported_at_count += count
                    file_reported_at += count

                # ------------------------------------------------
                # MISSING VALUES
                # ------------------------------------------------

                for column in chunk.columns:

                    missing = int(
                        chunk[column].isna().sum()
                    )

                    missing_counts[column] += missing

                # ------------------------------------------------
                # TRAIN + STATION OBSERVATIONS
                # ------------------------------------------------

                if (
                    "train_no" in chunk.columns
                    and "station_code" in chunk.columns
                ):

                    temp = chunk[
                        ["train_no", "station_code"]
                    ].copy()

                    temp = temp.dropna()

                    pairs = (
                        temp["train_no"]
                        .astype(str)
                        .str.strip()
                        + "|"
                        + temp["station_code"]
                        .astype(str)
                        .str.strip()
                    )

                    train_station_counts.update(pairs)

                # ------------------------------------------------
                # ROW DUPLICATE DETECTION
                # ------------------------------------------------

                # Hash a subset of important columns rather than
                # storing entire rows in RAM.
                hash_columns = [
                    c for c in [
                        "date",
                        "train_no",
                        "station_code",
                        "seq",
                        "sch_arrival",
                        "sch_departure",
                        "act_arrival",
                        "act_departure",
                        "distance_km",
                        "lat",
                        "lng",
                    ]
                    if c in chunk.columns
                ]

                if hash_columns:

                    subset = chunk[hash_columns].fillna("")

                    for row in subset.itertuples(
                        index=False,
                        name=None
                    ):

                        raw = "|".join(
                            str(x) for x in row
                        )

                        h = hashlib.md5(
                            raw.encode("utf-8")
                        ).hexdigest()

                        row_hash_counts[h] += 1

        except Exception as e:

            bad_files.append({
                "file": str(relative_path),
                "reason": f"Read error: {e}"
            })

            continue

        # --------------------------------------------------------
        # FILE STATISTICS
        # --------------------------------------------------------

        file_date_min = None
        file_date_max = None

        if file_dates:

            file_date_min = min(
                x[0] for x in file_dates
            )

            file_date_max = max(
                x[1] for x in file_dates
            )

        file_statistics.append({
            "file": str(relative_path),
            "size_mb": round(
                size_bytes / (1024 * 1024),
                3
            ),
            "rows": file_rows,
            "date_min": (
                file_date_min.strftime("%Y-%m-%d")
                if file_date_min is not None
                else None
            ),
            "date_max": (
                file_date_max.strftime("%Y-%m-%d")
                if file_date_max is not None
                else None
            ),
            "actual_arrival_rows": file_actual_arrival,
            "actual_departure_rows": file_actual_departure,
            "gps_rows": file_gps,
            "reported_at_rows": file_reported_at,
        })

        # --------------------------------------------------------
        # EXACT DUPLICATE FILE HASH
        # --------------------------------------------------------

        try:

            h = file_hash(filepath)

            file_hashes[h].append(
                str(relative_path)
            )

        except Exception:
            pass

    print("\n")

    # ============================================================
    # CALCULATE RESULTS
    # ============================================================

    duplicate_file_groups = {
        h: paths
        for h, paths in file_hashes.items()
        if len(paths) > 1
    }

    exact_duplicate_rows = sum(
        count - 1
        for count in row_hash_counts.values()
        if count > 1
    )

    total_size_gb = (
        total_size_bytes /
        (1024 ** 3)
    )

    def percentage(value):
        if total_rows == 0:
            return 0

        return round(
            value / total_rows * 100,
            3
        )

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Files                 : {len(files):,}")
    print(f"Total size            : {total_size_gb:.2f} GB")
    print(f"Total rows            : {total_rows:,}")
    print(f"Unique trains         : {len(unique_trains):,}")
    print(f"Unique stations       : {len(unique_stations):,}")
    print(f"Unique journeys       : {len(unique_journeys):,}")

    print()

    print(
        f"Date range            : "
        f"{date_min.date() if date_min else 'N/A'} "
        f"→ "
        f"{date_max.date() if date_max else 'N/A'}"
    )

    print()

    print(
        f"Actual arrival       : "
        f"{actual_arrival_count:,} "
        f"({percentage(actual_arrival_count)}%)"
    )

    print(
        f"Actual departure     : "
        f"{actual_departure_count:,} "
        f"({percentage(actual_departure_count)}%)"
    )

    print(
        f"GPS coordinates      : "
        f"{gps_count:,} "
        f"({percentage(gps_count)}%)"
    )

    print(
        f"Reported timestamps  : "
        f"{reported_at_count:,} "
        f"({percentage(reported_at_count)}%)"
    )

    print()

    print(
        f"Repeated row hashes  : "
        f"{exact_duplicate_rows:,}"
    )

    print(
        f"Duplicate file groups: "
        f"{len(duplicate_file_groups):,}"
    )

    print(
        f"Schema variants      : "
        f"{len(schema_counter):,}"
    )

    print(
        f"Problematic files    : "
        f"{len(bad_files):,}"
    )

    # ============================================================
    # SAVE SUMMARY JSON
    # ============================================================

    summary = {
        "dataset_directory": str(DATASET_DIR),
        "files": len(files),
        "total_size_gb": round(total_size_gb, 3),
        "total_rows": total_rows,

        "unique_trains": len(unique_trains),
        "unique_stations": len(unique_stations),
        "unique_journeys": len(unique_journeys),

        "date_min": (
            date_min.strftime("%Y-%m-%d")
            if date_min is not None
            else None
        ),

        "date_max": (
            date_max.strftime("%Y-%m-%d")
            if date_max is not None
            else None
        ),

        "actual_arrival_count": actual_arrival_count,
        "actual_arrival_percentage": percentage(
            actual_arrival_count
        ),

        "actual_departure_count": actual_departure_count,
        "actual_departure_percentage": percentage(
            actual_departure_count
        ),

        "gps_count": gps_count,
        "gps_percentage": percentage(gps_count),

        "reported_at_count": reported_at_count,
        "reported_at_percentage": percentage(
            reported_at_count
        ),

        "exact_duplicate_rows": exact_duplicate_rows,

        "duplicate_file_groups": len(
            duplicate_file_groups
        ),

        "schema_variants": len(schema_counter),

        "problematic_files": len(bad_files),

        "runtime_seconds": round(
            time.time() - start_time,
            2
        ),
    }

    with open(
        OUTPUT_DIR / "dataset_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            default=str
        )

    # ============================================================
    # SAVE FILE STATISTICS
    # ============================================================

    pd.DataFrame(
        file_statistics
    ).to_csv(
        OUTPUT_DIR / "file_statistics.csv",
        index=False
    )

    # ============================================================
    # SAVE MISSING VALUE REPORT
    # ============================================================

    missing_df = pd.DataFrame(
        [
            {
                "column": column,
                "missing_values": count,
                "missing_percentage": (
                    round(
                        count / total_rows * 100,
                        3
                    )
                    if total_rows
                    else 0
                ),
            }

            for column, count
            in missing_counts.items()
        ]
    )

    missing_df = missing_df.sort_values(
        "missing_percentage",
        ascending=False
    )

    missing_df.to_csv(
        OUTPUT_DIR / "missing_values.csv",
        index=False
    )

    # ============================================================
    # SAVE TRAIN STATISTICS
    # ============================================================

    train_df = pd.DataFrame(
        [
            {
                "train_no": train,
                "rows": count,
            }

            for train, count
            in train_counter.items()
        ]
    )

    train_df = train_df.sort_values(
        "rows",
        ascending=False
    )

    train_df.to_csv(
        OUTPUT_DIR / "train_statistics.csv",
        index=False
    )

    # ============================================================
    # SAVE STATION STATISTICS
    # ============================================================

    station_df = pd.DataFrame(
        [
            {
                "station_code": station,
                "rows": count,
            }

            for station, count
            in station_counter.items()
        ]
    )

    station_df = station_df.sort_values(
        "rows",
        ascending=False
    )

    station_df.to_csv(
        OUTPUT_DIR / "station_statistics.csv",
        index=False
    )

    # ============================================================
    # SAVE TRAIN-STATION FREQUENCY
    # ============================================================

    train_station_df = pd.DataFrame(
        [
            {
                "train_station": key,
                "observations": count,
            }

            for key, count
            in train_station_counts.items()
        ]
    )

    train_station_df = train_station_df.sort_values(
        "observations",
        ascending=False
    )

    train_station_df.to_csv(
        OUTPUT_DIR / "train_station_observations.csv",
        index=False
    )

    # ============================================================
    # SAVE SCHEMA REPORT
    # ============================================================

    schema_rows = []

    for schema, count in schema_counter.items():

        schema_rows.append({
            "file_count": count,
            "column_count": len(schema),
            "columns": " | ".join(schema),
        })

    pd.DataFrame(
        schema_rows
    ).to_csv(
        OUTPUT_DIR / "schema_report.csv",
        index=False
    )

    # ============================================================
    # SAVE BAD FILES
    # ============================================================

    pd.DataFrame(
        bad_files
    ).to_csv(
        OUTPUT_DIR / "problematic_files.csv",
        index=False
    )

    # ============================================================
    # SAVE DUPLICATE FILE GROUPS
    # ============================================================

    duplicate_rows = []

    for file_hash_value, paths in duplicate_file_groups.items():

        duplicate_rows.append({
            "md5": file_hash_value,
            "file_count": len(paths),
            "files": " | ".join(paths),
        })

    pd.DataFrame(
        duplicate_rows
    ).to_csv(
        OUTPUT_DIR / "duplicate_files.csv",
        index=False
    )

    # ============================================================
    # FINISHED
    # ============================================================

    runtime = time.time() - start_time

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"Runtime: {runtime / 60:.2f} minutes")
    print(f"Output : {OUTPUT_DIR.resolve()}")

    print()
    print("Files generated:")

    for path in sorted(OUTPUT_DIR.iterdir()):

        print(
            f"  - {path.name}"
        )

    print()
    print("Next step:")
    print(
        "Upload dataset_summary.json, "
        "missing_values.csv, schema_report.csv, "
        "file_statistics.csv, and problematic_files.csv."
    )


if __name__ == "__main__":
    main()