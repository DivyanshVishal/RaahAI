
from pathlib import Path
import duckdb

INPUT_DIR = Path(
    r"C:\System\ProjectV\processed\gps_data\year=2026"
)

REPORT_FILE = Path(
    r"C:\System\ProjectV\processed\raahai_2026\diagnostic_report_v3.txt"
)

MAX_FILES = 10

# ------------------------------------------------------------
# 1. Select the same sampling strategy as V2
# ------------------------------------------------------------

files = sorted(
    INPUT_DIR.rglob("*.parquet"),
    key=lambda p: p.stat().st_size,
    reverse=True
)

if not files:
    raise FileNotFoundError(f"No Parquet files found: {INPUT_DIR}")

selected = files[:MAX_FILES]
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit = '2GB'")
con.execute("SET threads = 2")

con.read_parquet(
    [str(p) for p in selected],
    union_by_name=True
).create_view("raw")

columns = {
    row[0]
    for row in con.execute("DESCRIBE SELECT * FROM raw").fetchall()
}

required = {
    "date", "train_no", "seq", "station_code",
    "distance_km", "act_arrival", "act_arrival_date",
    "reported_at"
}

missing = sorted(required - columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

report = [
    "RAILWAY DATA DIAGNOSTIC V3",
    f"Input directory: {INPUT_DIR}",
    f"Files examined: {len(selected)} of {len(files)}",
    "Sample is limited to the 10 largest files."
]


def add_section(title, query):
    report.extend([
        "",
        "=" * 72,
        title,
        "=" * 72
    ])
    result = con.execute(query).fetchdf()
    report.append(result.to_string(index=False))
    return result


# ------------------------------------------------------------
# 2. Normalize fields and parse timestamps
# ------------------------------------------------------------

con.execute("""
CREATE TEMP VIEW normalized AS
SELECT
    CAST(train_no AS VARCHAR) AS train_no,
    TRY_CAST(date AS DATE) AS journey_date,
    TRY_CAST(seq AS INTEGER) AS seq,
    UPPER(TRIM(CAST(station_code AS VARCHAR))) AS station_code,
    TRY_CAST(distance_km AS DOUBLE) AS distance_km,

    COALESCE(
        TRY_STRPTIME(
            CAST(act_arrival_date AS VARCHAR) || ' ' ||
            CAST(act_arrival AS VARCHAR),
            '%d %b %Y %H:%M:%S'
        ),
        TRY_STRPTIME(
            CAST(act_arrival_date AS VARCHAR) || ' ' ||
            CAST(act_arrival AS VARCHAR),
            '%d %b %Y %H:%M'
        ),
        TRY_CAST(
            CAST(act_arrival_date AS VARCHAR) || ' ' ||
            CAST(act_arrival AS VARCHAR) AS TIMESTAMP
        )
    ) AS actual_arrival_ts

FROM raw
WHERE train_no IS NOT NULL
  AND TRY_CAST(date AS DATE) IS NOT NULL
""")

# ------------------------------------------------------------
# 3. Distance variation for the same train and station
# ------------------------------------------------------------

add_section(
    "1. DISTANCE VARIATION FOR SAME TRAIN AND STATION",
    """
    WITH station_stats AS (
        SELECT
            train_no,
            station_code,
            COUNT(*) AS observations,
            COUNT(DISTINCT journey_date) AS journey_dates,
            MIN(distance_km) AS min_distance_km,
            MAX(distance_km) AS max_distance_km,
            AVG(distance_km) AS mean_distance_km,
            STDDEV_POP(distance_km) AS std_distance_km
        FROM normalized
        WHERE station_code IS NOT NULL
          AND distance_km IS NOT NULL
        GROUP BY train_no, station_code
        HAVING COUNT(DISTINCT journey_date) >= 3
    )
    SELECT
        *,
        max_distance_km - min_distance_km AS distance_range_km
    FROM station_stats
    ORDER BY distance_range_km DESC
    LIMIT 20
    """
)

add_section(
    "2. OVERALL DISTANCE STABILITY",
    """
    WITH station_stats AS (
        SELECT
            train_no,
            station_code,
            COUNT(DISTINCT journey_date) AS journey_dates,
            MIN(distance_km) AS min_km,
            MAX(distance_km) AS max_km
        FROM normalized
        WHERE station_code IS NOT NULL
          AND distance_km IS NOT NULL
        GROUP BY train_no, station_code
        HAVING COUNT(DISTINCT journey_date) >= 3
    )
    SELECT
        COUNT(*) AS train_station_pairs_checked,
        COUNT(*) FILTER (
            WHERE max_km - min_km <= 1
        ) AS pairs_with_range_at_most_1km,
        COUNT(*) FILTER (
            WHERE max_km - min_km > 1
        ) AS pairs_with_range_over_1km,
        COUNT(*) FILTER (
            WHERE max_km - min_km > 5
        ) AS pairs_with_range_over_5km,
        MAX(max_km - min_km) AS largest_distance_range_km
    FROM station_stats
    """
)

# ------------------------------------------------------------
# 4. Repeated sequence and station records within journeys
# ------------------------------------------------------------

add_section(
    "3. REPEATED SEQUENCE NUMBERS",
    """
    WITH seq_stats AS (
        SELECT
            train_no,
            journey_date,
            seq,
            COUNT(*) AS row_count,
            COUNT(DISTINCT station_code) AS stations_at_sequence
        FROM normalized
        WHERE seq IS NOT NULL
        GROUP BY train_no, journey_date, seq
    )
    SELECT
        COUNT(*) AS train_date_sequence_groups,
        COUNT(*) FILTER (
            WHERE row_count > 1
        ) AS repeated_sequence_groups,
        COUNT(*) FILTER (
            WHERE stations_at_sequence > 1
        ) AS sequences_with_multiple_stations,
        SUM(row_count - 1) FILTER (
            WHERE row_count > 1
        ) AS extra_rows_at_repeated_sequences
    FROM seq_stats
    """
)

add_section(
    "4. REPEATED STATION RECORDS WITHIN JOURNEYS",
    """
    WITH station_stats AS (
        SELECT
            train_no,
            journey_date,
            station_code,
            COUNT(*) AS row_count,
            COUNT(DISTINCT seq) AS distinct_sequences
        FROM normalized
        WHERE station_code IS NOT NULL
        GROUP BY train_no, journey_date, station_code
    )
    SELECT
        COUNT(*) AS train_date_station_groups,
        COUNT(*) FILTER (
            WHERE row_count > 1
        ) AS repeated_station_groups,
        COUNT(*) FILTER (
            WHERE distinct_sequences > 1
        ) AS stations_with_multiple_sequences,
        MAX(row_count) AS maximum_rows_for_station_in_journey
    FROM station_stats
    """
)

# ------------------------------------------------------------
# 5. Check arrival timestamps in station-sequence order
# ------------------------------------------------------------

con.execute("""
CREATE TEMP VIEW ordered_arrivals AS
SELECT
    *,
    LAG(actual_arrival_ts) OVER (
        PARTITION BY train_no, journey_date
        ORDER BY seq
    ) AS previous_arrival_ts
FROM normalized
WHERE seq IS NOT NULL
""")

add_section(
    "5. ACTUAL ARRIVAL TIMESTAMP ORDERING",
    """
    SELECT
        COUNT(*) FILTER (
            WHERE actual_arrival_ts IS NOT NULL
              AND previous_arrival_ts IS NOT NULL
        ) AS comparable_timestamp_pairs,

        COUNT(*) FILTER (
            WHERE actual_arrival_ts > previous_arrival_ts
        ) AS increasing_pairs,

        COUNT(*) FILTER (
            WHERE actual_arrival_ts = previous_arrival_ts
        ) AS equal_timestamp_pairs,

        COUNT(*) FILTER (
            WHERE actual_arrival_ts < previous_arrival_ts
        ) AS decreasing_timestamp_pairs,

        COUNT(*) FILTER (
            WHERE actual_arrival_ts < previous_arrival_ts
        ) * 100.0 /
        NULLIF(
            COUNT(*) FILTER (
                WHERE actual_arrival_ts IS NOT NULL
                  AND previous_arrival_ts IS NOT NULL
            ), 0
        ) AS decreasing_percent
    FROM ordered_arrivals
    """
)

add_section(
    "6. EXAMPLES OF DECREASING ARRIVAL TIMES",
    """
    SELECT
        train_no,
        journey_date,
        seq,
        station_code,
        previous_arrival_ts,
        actual_arrival_ts,
        DATE_DIFF(
            'minute',
            previous_arrival_ts,
            actual_arrival_ts
        ) AS minutes_since_previous_arrival
    FROM ordered_arrivals
    WHERE actual_arrival_ts IS NOT NULL
      AND previous_arrival_ts IS NOT NULL
      AND actual_arrival_ts < previous_arrival_ts
    ORDER BY minutes_since_previous_arrival
    LIMIT 20
    """
)

# ------------------------------------------------------------
# 6. Write report
# ------------------------------------------------------------

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("V3 diagnostic completed.")
print(f"Files examined: {len(selected)} of {len(files)}")
print(f"Report: {REPORT_FILE}")

con.close()
