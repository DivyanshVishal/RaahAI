
from pathlib import Path
import json
from datetime import datetime, timezone

import duckdb


# ============================================================
# CONFIGURATION — CHANGE THESE PATHS
# ============================================================

INPUT_DIR = Path(
    r"C:\System\ProjectV\processed\gps_data\year=2026"
)

OUTPUT_DIR = Path(
    r"C:\System\ProjectV\processed\raahai_2026\eta_features"
)

OUTPUT_FILE = OUTPUT_DIR / "railway_eta_features_2026.parquet"
SUMMARY_FILE = OUTPUT_DIR / "processing_summary.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set to False if you want to retain rows with invalid GPS.
# Rows without valid GPS cannot be used for GPS-based analysis.
REQUIRE_VALID_GPS = True


# ============================================================
# DISCOVER INPUT FILES
# ============================================================

parquet_files = sorted(INPUT_DIR.rglob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"No Parquet files found under: {INPUT_DIR}"
    )

print(f"Input directory : {INPUT_DIR}")
print(f"Parquet files   : {len(parquet_files):,}")
print("Initializing DuckDB...")


# ============================================================
# HELPERS
# ============================================================

def sql_string(value):
    """Safely quote a string literal for SQL."""
    return "'" + str(value).replace("'", "''") + "'"


def clean(column):
    """
    Normalize text values and treat common textual nulls
    as missing values.
    """
    value = f"LOWER(TRIM(CAST({column} AS VARCHAR)))"
    return f"""
        CASE
            WHEN {value} IN (
                '', 'nan', 'null', 'none', 'nat', '<na>'
            )
            THEN NULL
            ELSE TRIM(CAST({column} AS VARCHAR))
        END
    """


def safe_number(column):
    """Convert invalid numeric values into NULL."""
    return f"TRY_CAST({clean(column)} AS DOUBLE)"


def make_timestamp(date_column, time_column):
    """
    Combine date and time fields into a timestamp.
    Prefer the event-specific date, then fall back to
    the main journey date.
    """
    event_date = clean(date_column)
    event_time = clean(time_column)
    journey_date = clean("date")

    return f"""
        COALESCE(
            TRY_CAST(
                {event_date} || ' ' || {event_time}
                AS TIMESTAMP
            ),
            TRY_CAST(
                {journey_date} || ' ' || {event_time}
                AS TIMESTAMP
            )
        )
    """


# ============================================================
# CONNECT AND REGISTER PARQUET INPUT
# ============================================================

con = duckdb.connect()

# DuckDB reads Parquet directly without loading all files
# into a single Pandas DataFrame.
con.read_parquet(
    [str(p) for p in parquet_files],
    union_by_name=True
).create_view("raw")


# ============================================================
# STEP 1: CLEAN THE 24 SOURCE COLUMNS
# ============================================================

required_columns = [
    "date", "train_no", "train_name", "train_type", "seq",
    "station_code", "halts_here",
    "sch_arrival", "sch_arrival_date",
    "sch_departure", "sch_departure_date",
    "act_arrival", "act_arrival_date",
    "act_departure", "act_departure_date",
    "delay_arrival_min", "delay_departure_min",
    "platform", "distance_km", "lat", "lng",
    "departed", "reported", "reported_at",
]

actual_columns = {
    row[0] for row in con.execute(
        "DESCRIBE SELECT * FROM raw"
    ).fetchall()
}

missing_columns = sorted(set(required_columns) - actual_columns)

if missing_columns:
    raise ValueError(
        "Input schema is missing required columns: "
        + ", ".join(missing_columns)
    )

con.execute(f"""
CREATE TEMP VIEW cleaned AS
SELECT
    TRY_CAST({clean('date')} AS DATE) AS journey_date,

    {clean('train_no')} AS train_no,
    {clean('train_name')} AS train_name,
    {clean('train_type')} AS train_type,

    {safe_number('seq')} AS seq,
    {clean('station_code')} AS station_code,
    {clean('halts_here')} AS halts_here,

    {make_timestamp('sch_arrival_date', 'sch_arrival')}
        AS scheduled_arrival_ts,

    {make_timestamp('sch_departure_date', 'sch_departure')}
        AS scheduled_departure_ts,

    {make_timestamp('act_arrival_date', 'act_arrival')}
        AS actual_arrival_ts,

    {make_timestamp('act_departure_date', 'act_departure')}
        AS actual_departure_ts,

    {safe_number('delay_arrival_min')}
        AS delay_arrival_min,

    {safe_number('delay_departure_min')}
        AS delay_departure_min,

    {safe_number('distance_km')} AS distance_km,
    {safe_number('lat')} AS lat,
    {safe_number('lng')} AS lng,

    {clean('platform')} AS platform,
    {clean('departed')} AS departed,
    {clean('reported')} AS reported,

    TRY_CAST({clean('reported_at')} AS TIMESTAMP)
        AS reported_at_ts

FROM raw
""")


# ============================================================
# STEP 2: VALIDATE AND FILTER ROWS
# ============================================================

total_input_rows = con.execute(
    "SELECT COUNT(*) FROM cleaned"
).fetchone()[0]

invalid_gps_rows = con.execute("""
    SELECT COUNT(*)
    FROM cleaned
    WHERE lat IS NULL
       OR lng IS NULL
       OR lat NOT BETWEEN -90 AND 90
       OR lng NOT BETWEEN -180 AND 180
""").fetchone()[0]

invalid_date_rows = con.execute("""
    SELECT COUNT(*)
    FROM cleaned
    WHERE journey_date IS NULL
""").fetchone()[0]

con.execute("""
CREATE TEMP VIEW valid_rows AS
SELECT *
FROM cleaned
WHERE journey_date >= DATE '2026-01-01'
  AND journey_date <  DATE '2027-01-01'
  AND train_no IS NOT NULL
  AND station_code IS NOT NULL
  AND seq IS NOT NULL
  AND (
      NOT ?
      OR (
          lat IS NOT NULL
          AND lng IS NOT NULL
          AND lat BETWEEN -90 AND 90
          AND lng BETWEEN -180 AND 180
      )
  )
""", [REQUIRE_VALID_GPS])


# ============================================================
# STEP 3: REMOVE DUPLICATE OBSERVATIONS
# ============================================================

# DISTINCT uses all cleaned columns, not just the train/station
# key, so separate legitimate observations are retained.
con.execute("""
CREATE TEMP VIEW deduplicated AS
SELECT DISTINCT *
FROM valid_rows
""")

rows_before_dedup = con.execute(
    "SELECT COUNT(*) FROM valid_rows"
).fetchone()[0]

rows_after_dedup = con.execute(
    "SELECT COUNT(*) FROM deduplicated"
).fetchone()[0]

duplicates_removed = rows_before_dedup - rows_after_dedup


# ============================================================
# STEP 4: ORDER OBSERVATIONS AND DERIVE FEATURES
# ============================================================

# Assumption: journey_date + train_no identifies a journey.
# If your data contains multiple runs of the same train number
# on one journey date, add a genuine journey identifier here.

con.execute("""
CREATE TEMP VIEW ordered AS
SELECT
    *,

    FIRST_VALUE(station_code) OVER w AS origin_station,
    LAST_VALUE(station_code) OVER (
        PARTITION BY train_no, journey_date
        ORDER BY seq
        ROWS BETWEEN UNBOUNDED PRECEDING
                 AND UNBOUNDED FOLLOWING
    ) AS destination_station,

    LEAD(station_code) OVER w AS next_station_code,

    LEAD(distance_km) OVER w AS next_distance_km,
    FIRST_VALUE(distance_km) OVER w AS first_distance_km,
    LAST_VALUE(distance_km) OVER (
        PARTITION BY train_no, journey_date
        ORDER BY seq
        ROWS BETWEEN UNBOUNDED PRECEDING
                 AND UNBOUNDED FOLLOWING
    ) AS final_distance_km,

    LEAD(scheduled_arrival_ts) OVER w
        AS next_scheduled_arrival_ts,

    LAG(actual_departure_ts) OVER w
        AS previous_actual_departure_ts,

    LAG(actual_arrival_ts) OVER w
        AS previous_actual_arrival_ts,

    LAG(actual_departure_ts) OVER w
        AS previous_departure_for_segment,

    LAG(delay_arrival_min) OVER w
        AS previous_station_delay_min,

    ROW_NUMBER() OVER w AS station_position,

    COUNT(*) OVER (
        PARTITION BY train_no, journey_date
    ) AS total_stations

FROM deduplicated

WINDOW w AS (
    PARTITION BY train_no, journey_date
    ORDER BY seq
)
""")


# ============================================================
# STEP 5: CREATE ETA FEATURES AND TRAINING TARGETS
# ============================================================

con.execute("""
CREATE TEMP VIEW engineered AS
WITH features AS (
    SELECT
        journey_date AS date,
        train_no,
        train_type,

        station_code,
        next_station_code,
        origin_station,
        destination_station,
        seq,
        station_position,
        total_stations,

        lat,
        lng,
        halts_here,

        DAYOFWEEK(journey_date) AS day_of_week,
        MONTH(journey_date) AS month,

        COALESCE(
            HOUR(scheduled_departure_ts),
            HOUR(scheduled_arrival_ts)
        ) AS hour_of_day,

        scheduled_arrival_ts,
        scheduled_departure_ts,
        actual_arrival_ts,
        actual_departure_ts,

        -- Distances, assuming distance_km is cumulative
        distance_km - first_distance_km
            AS distance_from_origin_km,

        CASE
            WHEN next_distance_km >= distance_km
            THEN next_distance_km - distance_km
            ELSE NULL
        END AS distance_to_next_station_km,

        total_stations - station_position
            AS remaining_stations,

        CASE
            WHEN final_distance_km >= distance_km
            THEN final_distance_km - distance_km
            ELSE NULL
        END AS remaining_distance_km,

        -- Scheduled section and station durations
        DATE_DIFF(
            'second',
            scheduled_departure_ts,
            next_scheduled_arrival_ts
        ) / 60.0 AS scheduled_travel_time_min,

        DATE_DIFF(
            'second',
            scheduled_arrival_ts,
            scheduled_departure_ts
        ) / 60.0 AS scheduled_halt_min,

        -- Delay known at the current observation
        CASE
            WHEN actual_departure_ts IS NOT NULL
                THEN delay_departure_min
            WHEN actual_arrival_ts IS NOT NULL
                THEN delay_arrival_min
            ELSE COALESCE(
                delay_departure_min,
                delay_arrival_min
            )
        END AS latest_known_delay_min,

        reported_at_ts,

        -- Travel time into the current station
        DATE_DIFF(
            'second',
            previous_actual_departure_ts,
            actual_arrival_ts
        ) / 60.0 AS last_segment_travel_time_min,

        -- Completed halt at the previous station
        DATE_DIFF(
            'second',
            previous_actual_arrival_ts,
            previous_actual_departure_ts
        ) / 60.0 AS previous_halt_duration_min,

        previous_station_delay_min,

        -- TARGET: actual travel time from this station
        -- to the next station. Not an input feature.
        DATE_DIFF(
            'second',
            actual_departure_ts,
            LEAD(actual_arrival_ts) OVER (
                PARTITION BY train_no, date
                ORDER BY seq
            )
        ) / 60.0 AS actual_segment_travel_time_min,

        -- TARGET: actual halt at this station.
        -- Not an input feature for the same station.
        DATE_DIFF(
            'second',
            actual_arrival_ts,
            actual_departure_ts
        ) / 60.0 AS actual_halt_duration_min

    FROM ordered
)
SELECT
    *,

    -- Operational feature: recompute at prediction time.
    -- Do not blindly use this as a historical training feature.
    GREATEST(
        0,
        DATE_DIFF('second', reported_at_ts, CURRENT_TIMESTAMP)
        / 60.0
    ) AS minutes_since_last_update

FROM features
""")


# ============================================================
# STEP 6: SANITY CHECKS
# ============================================================

# Reject negative durations caused by inconsistent timestamps.
# Keep NULL values where the required timestamps are unavailable.
con.execute("""
CREATE TEMP VIEW final_features AS
SELECT
    * EXCLUDE (
        scheduled_travel_time_min,
        scheduled_halt_min,
        last_segment_travel_time_min,
        previous_halt_duration_min,
        actual_segment_travel_time_min,
        actual_halt_duration_min,
        distance_to_next_station_km,
        remaining_distance_km
    ),

    CASE WHEN scheduled_travel_time_min >= 0
         THEN scheduled_travel_time_min END
        AS scheduled_travel_time_min,

    CASE WHEN scheduled_halt_min >= 0
         THEN scheduled_halt_min END
        AS scheduled_halt_min,

    CASE WHEN last_segment_travel_time_min >= 0
         THEN last_segment_travel_time_min END
        AS last_segment_travel_time_min,

    CASE WHEN previous_halt_duration_min >= 0
         THEN previous_halt_duration_min END
        AS last_halt_duration_min,

    CASE WHEN actual_segment_travel_time_min >= 0
         THEN actual_segment_travel_time_min END
        AS actual_segment_travel_time_min,

    CASE WHEN actual_halt_duration_min >= 0
         THEN actual_halt_duration_min END
        AS actual_halt_duration_min,

    CASE WHEN distance_to_next_station_km >= 0
         THEN distance_to_next_station_km END
        AS distance_to_next_station_km,

    CASE WHEN remaining_distance_km >= 0
         THEN remaining_distance_km END
        AS remaining_distance_km

FROM engineered
""")


# ============================================================
# STEP 7: WRITE COMPRESSED PARQUET OUTPUT
# ============================================================

con.execute("""
COPY (
    SELECT *
    FROM final_features
    ORDER BY train_no, date, seq
)
TO ?
(FORMAT PARQUET, COMPRESSION ZSTD)
""", [str(OUTPUT_FILE)])

output_rows = con.execute(
    "SELECT COUNT(*) FROM final_features"
).fetchone()[0]


# ============================================================
# STEP 8: PROCESSING SUMMARY
# ============================================================

summary = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "input_directory": str(INPUT_DIR),
    "output_file": str(OUTPUT_FILE),
    "input_files": len(parquet_files),
    "input_rows": total_input_rows,
    "invalid_gps_rows_in_input": invalid_gps_rows,
    "missing_or_invalid_date_rows_in_input": invalid_date_rows,
    "rows_after_filtering": rows_before_dedup,
    "duplicate_rows_removed": duplicates_removed,
    "output_rows": output_rows,
    "output_columns": con.execute(
        "SELECT COUNT(*) FROM "
        "(DESCRIBE SELECT * FROM final_features)"
    ).fetchone()[0],
    "require_valid_gps": REQUIRE_VALID_GPS,
    "historical_features_created": False,
    "weather_features_created": False,
    "notes": [
        "Original Parquet files were not modified.",
        "Historical aggregates require a separate past-only calculation.",
        "Weather features require external weather data.",
        "Date parsing assumes event date and time fields are parseable.",
        "Distance calculations assume distance_km is cumulative.",
        "Journey grouping assumes train_no + date identifies a journey.",
        "Actual durations are targets, not valid inputs for the same future event."
    ]
}

SUMMARY_FILE.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"Input files              : {len(parquet_files):,}")
print(f"Input rows               : {total_input_rows:,}")
print(f"Invalid GPS rows         : {invalid_gps_rows:,}")
print(f"Invalid/missing dates    : {invalid_date_rows:,}")
print(f"Duplicates removed       : {duplicates_removed:,}")
print(f"Output rows              : {output_rows:,}")
print(f"Output file              : {OUTPUT_FILE}")
print(f"Summary                  : {SUMMARY_FILE}")

con.close()
