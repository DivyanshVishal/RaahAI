#!/usr/bin/env python3
"""
Train delay dataset cleaner
---------------------------
Cleans the raw station-level GPS dataset (one parquet file per train) and writes
a cleaned copy with the same file layout, plus a station coordinate lookup and a
cleaning log that counts what every rule did.

The raw files are never modified.

How it works (memory-safe: one small file per worker at a time)
    Pass 0  reads only date + train_no from every file to find partial days.
    Pass 1  cleans each file independently and writes it to the output folder.
    Finally builds stations.parquet (one coordinate per station) and the log.

Usage:
    python clean_train_data.py
    python clean_train_data.py processed/gps_data/year=2026 --out processed/clean_data/year=2026
    python clean_train_data.py --limit 200          # quick trial on 200 files
    python clean_train_data.py --workers 1          # no multiprocessing (debugging)

Outputs (in --out):
    <same relative paths as input>.parquet   cleaned rows
    _stations.parquet                        station_code -> stn_lat, stn_lng
    _cleaning_log.txt                        rule-by-rule counts and checks

Keep the output as parquet. Station codes such as NAT, NIL and NAN are real
codes, and reading a CSV with default settings can silently turn them into nulls.

Requires: pandas, numpy, pyarrow
"""
import argparse
import glob
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
DEFAULT_IN = "processed/gps_data/year=2026"
DEFAULT_OUT = "processed/clean_data/year=2026"

CFG = {
    # A journey date is treated as partial (and dropped) when it has fewer
    # journeys than this fraction of the median day.
    "min_day_coverage": 0.5,
    # Source delay and (actual - scheduled) may differ by this many minutes
    # before the row is flagged as a mismatch.
    "delay_mismatch_tol": 1.0,
    # Plausibility limits (minutes). Values outside are cleared, not dropped.
    "min_dep_delay": -15,     # leaving more than 15 min early is treated as an error
    "min_arr_delay": -180,    # arriving more than 3 h early is treated as an error
    "max_delay": 2880,        # more than 48 h late is treated as an error
    # Dwell longer than this (minutes) gets flagged, not removed.
    "long_dwell": 720,
    # One-day date errors: a jump between consecutive actual times within this many
    # minutes of +/-1440 (one day) is treated as a wrong date stamp and corrected.
    "day_shift_tol": 180,
}
DAY = 1440.0

# Duplicate labels for the same train category seen in the profile.
# Assumption: these pairs mean the same thing. Edit if your source says otherwise.
TRAIN_TYPE_MAP = {
    "MAIL/EXPRESS": "MEX",
    "SUPERFAST": "SUF",
    "EMU - KOLKATA": "EMU",
}

LOCAL_TZ = "Asia/Kolkata"
DT_FMT = "%d %b %Y %H:%M"       # e.g. '26 Jan 2026' + '10:45'
DATE_FMT = "%Y-%m-%d"           # e.g. '2026-01-25'
JKEY = ["date", "train_no"]     # one journey
PANDAS_2 = int(pd.__version__.split(".")[0]) >= 2

REQUIRED = [
    "date", "train_no", "train_name", "train_type", "seq", "station_code",
    "halts_here", "sch_arrival", "sch_arrival_date", "sch_departure",
    "sch_departure_date", "act_arrival", "act_arrival_date", "act_departure",
    "act_departure_date", "delay_arrival_min", "delay_departure_min", "platform",
    "distance_km", "lat", "lng", "departed", "reported", "reported_at",
]

# Plain-English description of every counter, in log order.
RULES = [
    ("rows_in", "Rows read"),
    ("drop_missing_key", "Rows dropped: missing date, train_no, seq or station_code"),
    ("drop_low_coverage_day", "Rows dropped: journey date is a partial day (see list below)"),
    ("drop_non_halting", "Rows dropped: halts_here = 0"),
    ("drop_duplicate_stop", "Rows dropped: repeated (date, train_no, seq); kept the latest report"),
    ("mask_unreported_arrival", "Actual arrival cleared: reported = 0, so the time was never observed"),
    ("mask_not_departed", "Actual departure cleared: departed = 0"),
    ("mask_origin_arrival", "Arrival cleared at seq 0 (origin): a train has no arrival there"),
    ("mask_terminal_departure", "Departure cleared at the train's last station: no departure there"),
    ("fix_day_late", "Actual times moved back one day (stamped a day late), arrivals + departures"),
    ("fix_day_early", "Actual times moved forward one day (stamped a day early)"),
    ("fix_day_journeys", "   journeys with a one-day correction"),
    ("mask_implausible_arr", "Actual arrival cleared: delay outside the plausible range"),
    ("mask_implausible_dep", "Actual departure cleared: delay outside the plausible range"),
    ("platform_invalid", "Platform cleared: value is not like 1, 12 or 2A"),
    ("unparsed_datetime", "Date/time values that could not be parsed (left empty)"),
    ("drop_order_violation_journeys", "Journeys dropped: times or distance go backwards between stops"),
    ("drop_order_violation_rows", "   rows in those journeys"),
    ("drop_short_journey_rows", "Rows dropped: journey has fewer than 2 stops left"),
    ("flag_delay_mismatch_arr", "Flagged (kept): source arrival delay differs from actual - scheduled"),
    ("flag_delay_mismatch_dep", "Flagged (kept): source departure delay differs from actual - scheduled"),
    ("flag_long_dwell", "Flagged (kept): dwell longer than the long-dwell limit"),
    ("flag_report_before_arrival", "Flagged (kept): reported_at earlier than actual arrival"),
    ("train_type_relabelled", "Rows whose train_type label was merged into a standard label"),
    ("rows_out", "Rows written"),
    ("journeys_out", "Journeys written"),
]

BUCKETS = [-np.inf, -5.5, -0.5, 0.5, 1.5, 5.5, 30.5, np.inf]
BUCKET_LABELS = ["<= -6", "-5..-1", "0", "1", "2..5", "6..30", "> 30"]


# ----------------------------------------------------------------- helpers
def as_str(s):
    """Any column -> trimmed pandas string with real NA for empty values."""
    s = s.astype("string").str.strip()
    return s.mask(s == "")


def to_bool(s):
    m = {"1": True, "1.0": True, "true": True, "yes": True,
         "0": False, "0.0": False, "false": False, "no": False}
    return as_str(s).str.lower().map(m).astype("boolean")


def is_false(s):
    return s.eq(False).fillna(False).astype(bool)


def _fallback_dt(x):
    kw = {"errors": "coerce"}
    if PANDAS_2:
        kw["format"] = "mixed"
    return pd.to_datetime(x, **kw)


def parse_date(s, c):
    out = pd.to_datetime(s, format=DATE_FMT, errors="coerce")
    left = s.notna() & out.isna()
    if left.any():
        out.loc[left] = _fallback_dt(s[left])
        c["unparsed_datetime"] += int((s.notna() & out.isna()).sum())
    return out


def combine_dt(d, t, c):
    combo = d + " " + t                      # NA if either part missing
    out = pd.to_datetime(combo, format=DT_FMT, errors="coerce")
    left = combo.notna() & out.isna()
    if left.any():
        out.loc[left] = _fallback_dt(combo[left])
        c["unparsed_datetime"] += int((combo.notna() & out.isna()).sum())
    return out


def parse_reported(s, c):
    kw = {"errors": "coerce", "utc": True}
    if PANDAS_2:
        kw["format"] = "ISO8601"
    r = pd.to_datetime(s, **kw).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    c["unparsed_datetime"] += int((s.notna() & r.isna()).sum())
    return r


def minutes(delta):
    return delta.dt.total_seconds() / 60.0


def norm_platform(p, c):
    q = p.str.upper().str.replace(r"\.0$", "", regex=True)
    ok = q.str.fullmatch(r"\d{1,2}[A-Z]?").fillna(False).astype(bool)
    c["platform_invalid"] += int((q.notna() & ~ok).sum())
    return q.where(ok)


def bucket(x):
    x = x.dropna()
    if not len(x):
        return {}
    return pd.cut(x, BUCKETS, labels=BUCKET_LABELS).value_counts().to_dict()


def fix_day_shifts(df, arr, dep, cfg, c, val):
    """
    Correct actual times stamped with the wrong date. Within a journey the actual
    events are read in order (arrival, departure, next arrival, ...). A jump of about
    +1 day between consecutive known delays starts a block stamped a day late; a jump
    of about -1 day ends it. Single delays of about -1 day (stamped a day early) are
    moved forward. Real delays of 21-27 hours appearing in one step are not plausible
    on a single section, so this only touches date errors.
    """
    n = len(df)
    if n == 0:
        return arr, dep, np.zeros(n, bool)
    tol = cfg["day_shift_tol"]
    jid = pd.factorize(df["date"].astype(str) + "|" + df["train_no"].astype(str))[0]
    E = np.empty(2 * n)
    E[0::2] = arr.to_numpy(dtype="float64")
    E[1::2] = dep.to_numpy(dtype="float64")
    J = np.repeat(jid, 2)
    valid = ~np.isnan(E)
    last = np.maximum.accumulate(np.where(valid, np.arange(2 * n), -1))
    prev = np.r_[-1, last[:-1]]
    has_prev = (prev >= 0) & (J[np.maximum(prev, 0)] == J)
    diff = np.where(has_prev, E - E[np.maximum(prev, 0)], np.nan)

    # size of large jumps, before any correction (shows whether a one-day peak exists)
    big = diff[np.abs(np.nan_to_num(diff)) >= 600]
    for lo_, hi_, lab in ((-3000, -1621, "<= -1621"), (-1620, -1260, "-1620..-1260 (about -1 day)"),
                          (-1259, -600, "-1259..-600"), (600, 1259, "600..1259"),
                          (1260, 1620, "1260..1620 (about +1 day)"), (1621, 3000, ">= 1621")):
        val[("jump", lab)] += int(((big >= lo_) & (big <= hi_)).sum())

    up = valid & has_prev & (np.abs(diff - DAY) <= tol) & (E - DAY >= cfg["min_arr_delay"])
    down = valid & has_prev & (np.abs(diff + DAY) <= tol)
    shift = np.zeros(2 * n)
    for j in np.unique(J[up]):                      # rare: loop only over affected journeys
        lo_, hi_ = np.searchsorted(J, j, "left"), np.searchsorted(J, j, "right")
        state = 0
        for k in range(lo_, hi_):
            if not valid[k]:
                continue
            if up[k]:
                state += 1
            elif down[k] and state > 0:
                state -= 1
            shift[k] = state * DAY
    early = valid & (shift == 0) & (E <= -(DAY - tol)) & (E + DAY >= cfg["min_arr_delay"]) \
        & (E + DAY <= cfg["max_delay"])
    shift[early] = -DAY

    c["fix_day_late"] += int((shift > 0).sum())
    c["fix_day_early"] += int((shift < 0).sum())
    rows = (shift[0::2] != 0) | (shift[1::2] != 0)
    if rows.any():
        c["fix_day_journeys"] += int(len(np.unique(jid[rows])))
    sa, sd = shift[0::2], shift[1::2]
    df["act_arr"] = df["act_arr"] - pd.to_timedelta(sa, unit="m")
    df["act_dep"] = df["act_dep"] - pd.to_timedelta(sd, unit="m")
    arr = pd.Series(arr.to_numpy(dtype="float64") - sa, index=arr.index)
    dep = pd.Series(dep.to_numpy(dtype="float64") - sd, index=dep.index)
    return arr, dep, rows


def drop(df, mask, name, c):
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    c[name] += int(mask.sum())
    return df[~mask.to_numpy()]


def wmedian(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, cw[-1] / 2.0)]


# ----------------------------------------------------------------- pass 0
def scan_dates(path):
    """Journeys per date in one file (reads two columns only)."""
    try:
        d = pd.read_parquet(path, columns=["date", "train_no"]).astype("string")
    except Exception:
        return None
    return d.drop_duplicates().groupby("date").size()


# ----------------------------------------------------------------- pass 1
def clean_file(task):
    path, rel, out_root, bad_dates, cfg = task
    c, val = Counter(), Counter()
    out_path = os.path.join(out_root, rel)

    try:
        raw = pd.read_parquet(path)
    except Exception as e:
        c["files_failed"] += 1
        return c, val, None, f"{rel}: read failed: {type(e).__name__}: {str(e)[:100]}"
    missing = [x for x in REQUIRED if x not in raw.columns]
    if missing:
        c["files_failed"] += 1
        return c, val, None, f"{rel}: missing columns {missing}"

    c["files_ok"] += 1
    c["rows_in"] += len(raw)
    s = {col: as_str(raw[col]) for col in REQUIRED}
    del raw

    # ---- parse into proper types
    df = pd.DataFrame({
        "date": parse_date(s["date"], c),
        "train_no": s["train_no"],
        "train_name_raw": s["train_name"],
        "train_type_raw": s["train_type"],
        "seq": pd.to_numeric(s["seq"], errors="coerce"),
        "station_code": s["station_code"],      # NAT / NIL / NAN are real codes: kept
        "halts_here": to_bool(s["halts_here"]),
        "sch_arr": combine_dt(s["sch_arrival_date"], s["sch_arrival"], c),
        "sch_dep": combine_dt(s["sch_departure_date"], s["sch_departure"], c),
        "act_arr": combine_dt(s["act_arrival_date"], s["act_arrival"], c),
        "act_dep": combine_dt(s["act_departure_date"], s["act_departure"], c),
        "src_delay_arr_min": pd.to_numeric(s["delay_arrival_min"], errors="coerce"),
        "src_delay_dep_min": pd.to_numeric(s["delay_departure_min"], errors="coerce"),
        "platform": norm_platform(s["platform"], c),
        "distance_km": pd.to_numeric(s["distance_km"], errors="coerce"),
        "gps_lat": pd.to_numeric(s["lat"], errors="coerce"),
        "gps_lng": pd.to_numeric(s["lng"], errors="coerce"),
        "departed": to_bool(s["departed"]),
        "reported": to_bool(s["reported"]),
        "reported_at": parse_reported(s["reported_at"], c),
    })
    del s

    # ---- row drops
    df = drop(df, df[["date", "train_no", "seq", "station_code"]].isna().any(axis=1),
              "drop_missing_key", c)
    if bad_dates:
        df = drop(df, df["date"].isin(bad_dates), "drop_low_coverage_day", c)
    df = drop(df, is_false(df["halts_here"]), "drop_non_halting", c)
    df = df.drop(columns="halts_here")
    df["seq"] = df["seq"].round().astype("Int64")

    df = df.sort_values(JKEY + ["seq", "reported_at"], na_position="first")
    df = drop(df, df.duplicated(JKEY + ["seq"], keep="last"), "drop_duplicate_stop", c)

    # ---- clear values that are not real observations
    m = is_false(df["reported"]) & df["act_arr"].notna()
    df.loc[m, "act_arr"] = pd.NaT
    c["mask_unreported_arrival"] += int(m.sum())

    m = is_false(df["departed"]) & df["act_dep"].notna()
    df.loc[m, "act_dep"] = pd.NaT
    c["mask_not_departed"] += int(m.sum())

    m = (df["seq"] == 0) & (df["sch_arr"].notna() | df["act_arr"].notna())
    df.loc[m, ["sch_arr", "act_arr"]] = pd.NaT
    c["mask_origin_arrival"] += int(m.sum())

    last_seq = df.groupby("train_no")["seq"].transform("max")
    m = (df["seq"] == last_seq) & (df["sch_dep"].notna() | df["act_dep"].notna())
    df.loc[m, ["sch_dep", "act_dep"]] = pd.NaT
    c["mask_terminal_departure"] += int(m.sum())

    # ---- delays recomputed from timestamps (authoritative)
    arr = minutes(df["act_arr"] - df["sch_arr"])
    dep = minutes(df["act_dep"] - df["sch_dep"])

    # Check: what does the computed delay look like where the source left it empty?
    for k, calc, src in (("arr", arr, df["src_delay_arr_min"]),
                         ("dep", dep, df["src_delay_dep_min"])):
        for b, n in bucket(calc[src.isna()]).items():
            val[(k, "source delay EMPTY", b)] += int(n)
        for b, n in bucket(calc[src.notna()]).items():
            val[(k, "source delay present", b)] += int(n)

    tol = cfg["delay_mismatch_tol"]
    for k, calc, src in (("arr", arr, df["src_delay_arr_min"]),
                         ("dep", dep, df["src_delay_dep_min"])):
        mism = (calc.notna() & src.notna() & ((calc - src).abs() > tol)).astype(bool)
        df[f"flag_delay_mismatch_{k}"] = mism.to_numpy()
        c[f"flag_delay_mismatch_{k}"] += int(mism.sum())

    # ---- one-day date errors (before the plausibility limits, which they would trip)
    arr, dep, shifted = fix_day_shifts(df, arr, dep, cfg, c, val)
    df["flag_day_shift"] = shifted

    bad = arr.notna() & ((arr < cfg["min_arr_delay"]) | (arr > cfg["max_delay"]))
    df.loc[bad, "act_arr"] = pd.NaT
    arr[bad] = np.nan
    c["mask_implausible_arr"] += int(bad.sum())

    bad = dep.notna() & ((dep < cfg["min_dep_delay"]) | (dep > cfg["max_delay"]))
    df.loc[bad, "act_dep"] = pd.NaT
    dep[bad] = np.nan
    c["mask_implausible_dep"] += int(bad.sum())

    df["delay_arr_min"] = arr.round()
    df["delay_dep_min"] = dep.round()

    # ---- journey-level checks: drop journeys where time or distance runs backwards
    df = df.sort_values(JKEY + ["seq"]).reset_index(drop=True)
    g = df.groupby(JKEY, sort=False)
    nxt_sch_arr = g["sch_arr"].shift(-1)
    nxt_act_arr = g["act_arr"].shift(-1)
    nxt_dist = g["distance_km"].shift(-1)
    viol = ((df["sch_dep"] > nxt_sch_arr) | (df["act_dep"] > nxt_act_arr)
            | (df["sch_dep"] < df["sch_arr"]) | (df["act_dep"] < df["act_arr"])
            | (nxt_dist < df["distance_km"])).fillna(False).astype(bool)
    bad_j = viol.groupby([df["date"], df["train_no"]]).transform("any").astype(bool)
    if bad_j.any():
        c["drop_order_violation_journeys"] += int(df.loc[bad_j, JKEY].drop_duplicates().shape[0])
    df = drop(df, bad_j, "drop_order_violation_rows", c)

    size = df.groupby(JKEY)["seq"].transform("size")
    df = drop(df, size < 2, "drop_short_journey_rows", c)

    if df.empty:
        c["files_empty_after_cleaning"] += 1
        if os.path.exists(out_path):
            os.remove(out_path)                 # don't leave a stale file from an older run
        return c, val, None, None

    # ---- flags and journey context
    df = df.reset_index(drop=True)
    g = df.groupby(JKEY, sort=False)
    df["stop_idx"] = g.cumcount().astype("int32")
    df["n_stops"] = g["seq"].transform("size").astype("int32")
    df["first_seq"] = g["seq"].transform("min")

    sch_dwell = minutes(df["sch_dep"] - df["sch_arr"])
    act_dwell = minutes(df["act_dep"] - df["act_arr"])
    df["flag_long_dwell"] = ((sch_dwell > cfg["long_dwell"]) |
                             (act_dwell > cfg["long_dwell"])).fillna(False).astype(bool)
    c["flag_long_dwell"] += int(df["flag_long_dwell"].sum())

    early = minutes(df["reported_at"] - df["act_arr"]) < -1
    df["flag_report_before_arrival"] = early.fillna(False).astype(bool)
    c["flag_report_before_arrival"] += int(df["flag_report_before_arrival"].sum())

    # ---- train labels
    df["train_type"] = df["train_type_raw"].replace(TRAIN_TYPE_MAP)
    c["train_type_relabelled"] += int(df["train_type_raw"].isin(TRAIN_TYPE_MAP.keys()).sum())
    name_mode = (df.dropna(subset=["train_name_raw"])
                 .groupby("train_no")["train_name_raw"]
                 .agg(lambda x: x.value_counts().index[0]))
    df["train_name"] = df["train_no"].map(name_mode).astype("string")

    # ---- station coordinates seen in this file (lat/lng are train GPS fixes)
    st = (df.dropna(subset=["gps_lat", "gps_lng"])
          .groupby("station_code")
          .agg(lat=("gps_lat", "median"), lng=("gps_lng", "median"), n=("gps_lat", "size"))
          .reset_index())

    cols = [
        "date", "train_no", "train_name", "train_type", "train_type_raw",
        "seq", "stop_idx", "n_stops", "first_seq", "station_code", "platform",
        "sch_arr", "sch_dep", "act_arr", "act_dep",
        "delay_arr_min", "delay_dep_min", "src_delay_arr_min", "src_delay_dep_min",
        "distance_km", "gps_lat", "gps_lng", "reported", "departed", "reported_at",
        "flag_delay_mismatch_arr", "flag_delay_mismatch_dep",
        "flag_long_dwell", "flag_report_before_arrival", "flag_day_shift",
    ]
    df = df[cols]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)

    c["rows_out"] += len(df)
    c["journeys_out"] += int(df[JKEY].drop_duplicates().shape[0])
    c["files_written"] += 1
    return c, val, st, None


# ----------------------------------------------------------------- runner
def run_map(func, items, workers, chunksize):
    if workers <= 1:
        for it in items:
            yield func(it)
    else:
        with Pool(workers) as pool:
            for r in pool.imap_unordered(func, items, chunksize=chunksize):
                yield r


def main():
    ap = argparse.ArgumentParser(description="Clean the train delay dataset.")
    ap.add_argument("path", nargs="?", default=DEFAULT_IN, help="input folder of parquet files")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--limit", type=int, default=0, help="only process the first N files (trial run)")
    ap.add_argument("--min-day-coverage", type=float, default=CFG["min_day_coverage"])
    args = ap.parse_args()
    cfg = dict(CFG, min_day_coverage=args.min_day_coverage)

    in_dir, out_dir = os.path.abspath(args.path), os.path.abspath(args.out)
    if in_dir == out_dir or out_dir.startswith(in_dir + os.sep):
        sys.exit("Output folder must be outside the input folder.")
    files = sorted(glob.glob(os.path.join(in_dir, "**", "*.parquet"), recursive=True))
    if not files:
        sys.exit(f"No .parquet files found under {in_dir}")
    if args.limit:
        files = files[:args.limit]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"Files: {len(files):,}  |  workers: {args.workers}  |  output: {out_dir}")

    # ---- pass 0: journeys per date
    print("Pass 0: counting journeys per date ...")
    per_date = None
    for r in run_map(scan_dates, files, args.workers, 64):
        if r is not None:
            per_date = r if per_date is None else per_date.add(r, fill_value=0)
    per_date = per_date.sort_index()
    threshold = cfg["min_day_coverage"] * per_date.median()
    low = per_date[per_date < threshold]
    bad_dates = set(pd.to_datetime(low.index, format=DATE_FMT, errors="coerce").dropna())
    print(f"  {len(per_date)} dates, median {per_date.median():.0f} journeys/day; "
          f"{len(low)} partial day(s) will be dropped: {list(low.index)}")

    # ---- pass 1: clean every file
    print("Pass 1: cleaning files ...")
    tasks = [(f, os.path.relpath(f, in_dir), out_dir, bad_dates, cfg) for f in files]
    C, V, stations, errors = Counter(), Counter(), [], []
    for i, (c, val, st, err) in enumerate(run_map(clean_file, tasks, args.workers, 8), 1):
        C.update(c)
        V.update(val)
        if st is not None:
            stations.append(st)
        if err:
            errors.append(err)
        if i % 250 == 0 or i == len(tasks):
            el = time.time() - t0
            print(f"  [{i:>6,}/{len(tasks):,}] rows in {C['rows_in']:>12,}  out {C['rows_out']:>12,}  "
                  f"failed {C['files_failed']}  ({el:,.0f}s)")

    # ---- station lookup: weighted median of per-file medians
    st_out = os.path.join(out_dir, "_stations.parquet")
    if stations:
        allst = pd.concat(stations, ignore_index=True)
        rows = []
        for code, grp in allst.groupby("station_code"):
            rows.append({"station_code": code,
                         "stn_lat": float(wmedian(grp["lat"].to_numpy(), grp["n"].to_numpy())),
                         "stn_lng": float(wmedian(grp["lng"].to_numpy(), grp["n"].to_numpy())),
                         "n_obs": int(grp["n"].sum())})
        stn = pd.DataFrame(rows)
        stn.to_parquet(st_out, index=False)
    else:
        stn = pd.DataFrame()

    # ---- log
    L = []
    L += ["=" * 78, "CLEANING LOG", "=" * 78,
          f"Input : {in_dir}", f"Output: {out_dir}",
          f"Files : {len(files):,} processed | {C['files_written']:,} written | "
          f"{C['files_failed']} failed | {C['files_empty_after_cleaning']} empty after cleaning",
          f"Time  : {time.time() - t0:,.0f}s", "",
          "Settings: " + ", ".join(f"{k}={v}" for k, v in cfg.items()),
          "Train type merges: " + ", ".join(f"{k} -> {v}" for k, v in TRAIN_TYPE_MAP.items()),
          "", "-" * 78, "WHAT EACH RULE DID", "-" * 78]
    rin = max(C["rows_in"], 1)
    for key, desc in RULES:
        n = C[key]
        pct = f"{100 * n / rin:6.2f}%" if key not in ("journeys_out", "drop_order_violation_journeys", "fix_day_journeys") else "       "
        L.append(f"{n:>12,}  {pct}  {desc}")

    L += ["", "-" * 78, "PARTIAL DAYS DROPPED", "-" * 78,
          f"Median journeys/day {per_date.median():.0f}; threshold {threshold:.0f}"]
    L += [f"  {d}: {int(n):,} journeys" for d, n in low.items()] or ["  none"]

    L += ["", "-" * 78, "CHECK: COMPUTED DELAY WHERE THE SOURCE DELAY WAS EMPTY vs PRESENT", "-" * 78,
          "If 'source delay EMPTY' sits almost entirely in 0 / 1 / early buckets, the source",
          "left on-time stops blank, and the recomputed delay correctly fills them in.", ""]
    for k in ("arr", "dep"):
        for kind in ("source delay EMPTY", "source delay present"):
            tot = sum(V[(k, kind, b)] for b in BUCKET_LABELS)
            if not tot:
                continue
            parts = "  ".join(f"{b}: {100 * V[(k, kind, b)] / tot:5.1f}%" for b in BUCKET_LABELS)
            L.append(f"{k} | {kind:<21} n={tot:>10,} | {parts}")

    L += ["", "-" * 78, "LARGE JUMPS BETWEEN CONSECUTIVE ACTUAL TIMES (before correction, minutes)", "-" * 78,
          "A clear peak at about +/-1 day compared with the neighbouring buckets means some times",
          "were stamped with the wrong date; those are the ones corrected above.", ""]
    for lab in ("<= -1621", "-1620..-1260 (about -1 day)", "-1259..-600", "600..1259",
                "1260..1620 (about +1 day)", ">= 1621"):
        L.append(f"  {lab:<30} {V[('jump', lab)]:>10,}")

    if len(stn):
        L += ["", "-" * 78, "STATIONS", "-" * 78,
              f"{len(stn):,} stations written to _stations.parquet "
              f"(coordinate = weighted median of train GPS fixes reported at that station)",
              f"Stations with fewer than 5 GPS fixes: {int((stn['n_obs'] < 5).sum()):,}"]

    if errors:
        L += ["", "-" * 78, f"ERRORS ({len(errors)})", "-" * 78] + errors[:30]
        if len(errors) > 30:
            L.append(f"... {len(errors) - 30} more")

    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "_cleaning_log.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved log to {os.path.join(out_dir, '_cleaning_log.txt')}")


if __name__ == "__main__":
    freeze_support()
    main()
