from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import time
import re


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS
INPUT_DIR = Path(r"C:\System\ProjectV\ETADataset\trains")

# Output directory
OUTPUT_DIR = Path(
    r"C:\System\ProjectV\processed\gps_data"
)

# Temporary/stats directory
STATS_DIR = Path(
    r"C:\System\ProjectV\processed\gps_dataset_stats"
)

# Number of rows processed at once
CHUNK_SIZE = 100_000


# ============================================================
# COLUMNS WE WANT TO KEEP
# ============================================================

KEEP_COLUMNS = [
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
# COLUMN CLEANING
# ============================================================

def clean_column_name(col):
    """
    Normalize malformed column names.
    """

    col = str(col).strip()

    # Markdown artifacts
    col = col.replace("**", "")

    # HTML entity artifacts seen in the source data
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
    return [
        clean_column_name(c)
        for c in columns
    ]


# ============================================================
# NUMERIC CLEANING
# ============================================================

def clean_numeric(df, columns):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# COORDINATE VALIDATION
# ============================================================

def filter_valid_coordinates(df):

    # Convert coordinates to numeric
    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lng"] = pd.to_numeric(
        df["lng"],
        errors="coerce"
    )

    # Keep only non-null coordinates
    valid = (
        df["lat"].notna()
        &
        df["lng"].notna()
    )

    # Valid latitude range
    valid &= (
        df["lat"].between(-90, 90)
    )

    # Valid longitude range
    valid &= (
        df["lng"].between(-180, 180)
    )

    df = df.loc[valid].copy()

    return df


# ============================================================
# STRING CLEANING
# ============================================================

def clean_strings(df):

    string_columns = [
        "train_no",
        "train_name",
        "train_type",
        "station_code",
        "platform",
        "sch_arrival",
        "sch_arrival_date",
        "sch_departure",
        "sch_departure_date",
        "act_arrival",
        "act_arrival_date",
        "act_departure",
        "act_departure_date",
        "reported_at",
    ]

    for column in string_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

            # Convert common textual nulls
            df[column] = df[column].replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "NaN": pd.NA,
                    "NULL": pd.NA,
                    "null": pd.NA,
                    "None": pd.NA,
                }
            )

    return df


# ============================================================
# DATE CLEANING
# ============================================================

def clean_dates(df):

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    # These are kept as strings for now.
    #
    # We will later construct proper event timestamps
    # from *_date + *_time fields.
    #
    # This is intentional because railway schedules can
    # cross midnight.

    return df


# ============================================================
# EXACT DUPLICATE REMOVAL
# ============================================================

def remove_exact_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    return df, removed


# ============================================================
# OUTPUT COLUMN ORDER
# ============================================================

def reorder_columns(df):

    existing = [
        c for c in KEEP_COLUMNS
        if c in df.columns
    ]

    return df[existing]


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("RAAHAI GPS DATASET BUILDER")
    print("=" * 70)

    if not INPUT_DIR.exists():

        print(
            f"\nERROR: Input directory does not exist:\n"
            f"{INPUT_DIR}"
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    STATS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------

    files = sorted(
        INPUT_DIR.rglob("*.csv")
    )

    print(
        f"\nInput files : {len(files):,}"
    )

    # --------------------------------------------------------
    # GLOBAL STATISTICS
    # --------------------------------------------------------

    total_input_rows = 0
    total_gps_rows = 0
    total_output_rows = 0

    total_invalid_coordinates = 0
    total_exact_duplicates = 0

    files_processed = 0
    files_failed = 0

    failed_files = []

    yearly_counts = {}

    # --------------------------------------------------------
    # PROCESS EACH FILE
    # --------------------------------------------------------

    for file_index, filepath in enumerate(
        files,
        start=1
    ):

        print(
            f"\rProcessing "
            f"{file_index:,}/{len(files):,} "
            f"| output rows: "
            f"{total_output_rows:,}",
            end="",
            flush=True
        )

        try:

            for chunk in pd.read_csv(
                filepath,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ):

                # ------------------------------------------------
                # NORMALIZE COLUMN NAMES
                # ------------------------------------------------

                chunk.columns = normalize_columns(
                    chunk.columns
                )

                rows_before = len(chunk)

                total_input_rows += rows_before

                # ------------------------------------------------
                # CHECK REQUIRED COLUMNS
                # ------------------------------------------------

                required = {
                    "lat",
                    "lng"
                }

                if not required.issubset(
                    chunk.columns
                ):

                    raise ValueError(
                        "Missing lat/lng columns"
                    )

                # ------------------------------------------------
                # KEEP RELEVANT COLUMNS
                # ------------------------------------------------

                available_columns = [
                    c
                    for c in KEEP_COLUMNS
                    if c in chunk.columns
                ]

                chunk = chunk[
                    available_columns
                ].copy()

                # ------------------------------------------------
                # CLEAN STRINGS
                # ------------------------------------------------

                chunk = clean_strings(
                    chunk
                )

                # ------------------------------------------------
                # CLEAN NUMERICS
                # ------------------------------------------------

                numeric_columns = [
                    "seq",
                    "delay_arrival_min",
                    "delay_departure_min",
                    "distance_km",
                    "lat",
                    "lng",
                ]

                chunk = clean_numeric(
                    chunk,
                    numeric_columns
                )

                # ------------------------------------------------
                # CLEAN DATE
                # ------------------------------------------------

                chunk = clean_dates(
                    chunk
                )

                # ------------------------------------------------
                # GPS FILTER
                # ------------------------------------------------

                before_gps = len(chunk)

                chunk = filter_valid_coordinates(
                    chunk
                )

                gps_rows = len(chunk)

                total_gps_rows += gps_rows

                total_invalid_coordinates += (
                    before_gps - gps_rows
                )

                # Nothing left
                if chunk.empty:
                    continue

                # ------------------------------------------------
                # EXACT DUPLICATES
                # ------------------------------------------------

                chunk, duplicates = (
                    remove_exact_duplicates(
                        chunk
                    )
                )

                total_exact_duplicates += (
                    duplicates
                )

                if chunk.empty:
                    continue

                # ------------------------------------------------
                # CREATE YEAR PARTITION
                # ------------------------------------------------

                if "date" not in chunk.columns:
                    continue

                chunk = chunk[
                    chunk["date"].notna()
                ].copy()

                if chunk.empty:
                    continue

                chunk["year"] = (
                    chunk["date"]
                    .dt.year
                    .astype("int16")
                )

                # ------------------------------------------------
                # WRITE EACH YEAR SEPARATELY
                # ------------------------------------------------

                for year, year_df in chunk.groupby(
                    "year"
                ):

                    year = int(year)

                    year_dir = (
                        OUTPUT_DIR
                        / f"year={year}"
                    )

                    year_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    # Create a unique part filename
                    output_file = (
                        year_dir
                        / (
                            f"{filepath.stem}_"
                            f"{file_index}.parquet"
                        )
                    )

                    # Remove helper column
                    year_df = year_df.drop(
                        columns=["year"]
                    )

                    # Ensure column order
                    year_df = reorder_columns(
                        year_df
                    )

                    # ------------------------------------------------
                    # WRITE PARQUET
                    # ------------------------------------------------

                    table = pa.Table.from_pandas(
                        year_df,
                        preserve_index=False
                    )

                    pq.write_table(
                        table,
                        output_file,
                        compression="zstd"
                    )

                    count = len(year_df)

                    total_output_rows += count

                    yearly_counts[year] = (
                        yearly_counts.get(
                            year,
                            0
                        )
                        + count
                    )

            files_processed += 1

        except Exception as e:

            files_failed += 1

            failed_files.append({
                "file": str(filepath),
                "error": str(e),
            })

    print("\n")

    # ============================================================
    # SAVE STATISTICS
    # ============================================================

    summary = {

        "input_directory": str(
            INPUT_DIR
        ),

        "output_directory": str(
            OUTPUT_DIR
        ),

        "input_files": len(files),

        "files_processed": (
            files_processed
        ),

        "files_failed": (
            files_failed
        ),

        "input_rows": (
            total_input_rows
        ),

        "rows_with_valid_gps": (
            total_gps_rows
        ),

        "rows_written": (
            total_output_rows
        ),

        "invalid_coordinate_rows": (
            total_invalid_coordinates
        ),

        "exact_duplicates_removed": (
            total_exact_duplicates
        ),

        "yearly_row_counts": (
            yearly_counts
        ),

        "runtime_seconds": round(
            time.time() - start_time,
            2
        ),
    }

    with open(
        STATS_DIR / "processing_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # FAILED FILES
    # --------------------------------------------------------

    if failed_files:

        pd.DataFrame(
            failed_files
        ).to_csv(
            STATS_DIR / "failed_files.csv",
            index=False
        )

    # ============================================================
    # FINAL REPORT
    # ============================================================

    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nInput files             : "
        f"{len(files):,}"
    )

    print(
        f"Successfully processed  : "
        f"{files_processed:,}"
    )

    print(
        f"Failed files            : "
        f"{files_failed:,}"
    )

    print(
        f"\nInput rows              : "
        f"{total_input_rows:,}"
    )

    print(
        f"Rows with valid GPS     : "
        f"{total_gps_rows:,}"
    )

    print(
        f"Rows written            : "
        f"{total_output_rows:,}"
    )

    print(
        f"Invalid GPS rows        : "
        f"{total_invalid_coordinates:,}"
    )

    print(
        f"Exact duplicates removed: "
        f"{total_exact_duplicates:,}"
    )

    print("\nRows by year:")

    for year in sorted(
        yearly_counts
    ):

        print(
            f"  {year}: "
            f"{yearly_counts[year]:,}"
        )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        f"\nRuntime: "
        f"{time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()