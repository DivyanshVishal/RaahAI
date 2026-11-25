#!/usr/bin/env python3
"""
Segment builder (stage 1 of feature engineering)
------------------------------------------------
Turns the cleaned station-level rows into station-to-station SEGMENT rows:
one row per (journey date, train, stop A -> next stop B).

Adds every feature that can be computed from a single train's own data:
    targets          arrival delay at B, and delay gained between A and B
    schedule         planned run time, distance, planned speed, stations passed
    live state       delay when leaving A, delay gained on recent segments
    position         where the segment sits in the journey
    calendar         hour (cyclic), day of week, month, journey day
    stations         coordinates of A, B and the midpoint (for joining weather)
    history          this train on this segment over the PREVIOUS 7/30/90 days
    quality flags    e.g. arrival at B was back-calculated, not observed

Features that need ALL trains at once (section congestion, station busyness,
segment difficulty across trains, holidays, weather) come in stage 2.

Each output file is one train, same layout as the input. Raw and cleaned
files are never modified.

Usage:
    python build_segments.py
    python build_segments.py processed/clean_data/year=2026 --out processed/segments/year=2026
    python build_segments.py --limit 200        # quick trial
    python build_segments.py --workers 1        # no multiprocessing (debugging)

Outputs (in --out):
    <same relative paths>.parquet   segment rows
    _segments_log.txt               counts, checks, target distribution
    _data_dictionary.csv            every column, its group, and when it is known

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

DEFAULT_IN = "processed/clean_data/year=2026"
DEFAULT_OUT = "processed/segments/year=2026"

CFG = {
    "max_speed_kmph": 180,   # planned or actual speed above this gets flagged
    "hist_windows": (7, 30, 90),
}
JKEY = ["date", "train_no"]
SEG_KEY = ["train_no", "from_stn", "to_stn"]

# ---------------------------------------------------------------- dictionary
# (column, group, known_when, description)
#   known_when:  schedule = known any time in advance
#                live     = known once the train has left station A
#                history  = uses only days BEFORE the journey date
#                target   = the label to predict; never use as an input
#                quality  = for filtering / weighting / evaluation, not an input
DICTIONARY = [
    ("date", "key", "schedule", "Journey start date"),
    ("train_no", "key", "schedule", "Train number"),
    ("train_type", "train", "schedule", "Train category after label merges (EMU, MEX, PAS, ...)"),
    ("from_stn", "key", "schedule", "Station A (segment start)"),
    ("to_stn", "key", "schedule", "Station B (segment end)"),
    ("from_seq", "segment", "schedule", "Route sequence number of A"),
    ("to_seq", "segment", "schedule", "Route sequence number of B"),
    ("seg_idx", "position", "schedule", "Segment number within this journey (0 = first)"),
    ("n_segments", "position", "schedule", "Segments in this journey"),
    ("frac_journey", "position", "schedule", "seg_idx / n_segments (0 = start, 1 = end)"),
    ("is_first_segment", "position", "schedule", "True if A is the journey's first stop"),
    ("is_last_segment", "position", "schedule", "True if B is the journey's last stop"),
    ("journey_day", "position", "schedule", "Day of journey when leaving A (0 = start date)"),
    ("km_A", "segment", "schedule", "Route distance of A from the route origin (km)"),
    ("seg_km", "segment", "schedule", "Distance A -> B (km)"),
    ("km_to_end", "position", "schedule", "Distance from B to the journey's last stop (km)"),
    ("stations_passed", "segment", "schedule", "Route stations between A and B where the train does not stop"),
    ("sch_dep_A", "schedule", "schedule", "Scheduled departure from A"),
    ("sch_arr_B", "schedule", "schedule", "Scheduled arrival at B"),
    ("sch_run_min", "schedule", "schedule", "Scheduled running time A -> B (min)"),
    ("sch_speed_kmph", "schedule", "schedule", "Scheduled average speed A -> B (km/h)"),
    ("sch_dwell_A_min", "schedule", "schedule", "Scheduled stop time at A (min); empty at the journey's first stop"),
    ("sch_dwell_B_min", "schedule", "schedule", "Scheduled stop time at B (min); empty at the last stop"),
    ("dep_hour", "calendar", "schedule", "Scheduled departure time from A as decimal hour (e.g. 13.5)"),
    ("hour_sin", "calendar", "schedule", "sin of departure hour (cyclic encoding)"),
    ("hour_cos", "calendar", "schedule", "cos of departure hour (cyclic encoding)"),
    ("dow", "calendar", "schedule", "Day of week of departure from A (0 = Monday)"),
    ("is_weekend", "calendar", "schedule", "Saturday or Sunday"),
    ("month", "calendar", "schedule", "Month of departure from A"),
    ("day_of_year", "calendar", "schedule", "Day of year of departure from A"),
    ("from_lat", "station", "schedule", "Latitude of A (station lookup)"),
    ("from_lng", "station", "schedule", "Longitude of A"),
    ("to_lat", "station", "schedule", "Latitude of B"),
    ("to_lng", "station", "schedule", "Longitude of B"),
    ("mid_lat", "station", "schedule", "Segment midpoint latitude (for weather joins)"),
    ("mid_lng", "station", "schedule", "Segment midpoint longitude"),
    ("dep_delay_A", "live", "live", "Departure delay at A (min). The strongest single predictor"),
    ("arr_delay_A", "live", "live", "Arrival delay at A (min); empty at the journey's first stop"),
    ("dwell_overrun_A", "live", "live", "Actual minus scheduled stop time at A (min)"),
    ("gain_prev1", "live", "live", "Delay gained on the segment that ended at A (min)"),
    ("gain_prev2", "live", "live", "Delay gained on the segment before that (min)"),
    ("trend_2stops", "live", "live", "Departure delay at A minus departure delay two stops earlier (min)"),
    ("origin_dep_delay", "live", "live", "Departure delay at the journey's first stop (min)"),
    ("max_dep_delay_so_far", "live", "live", "Largest departure delay so far in this journey (min)"),
    ("arr_imputed_A", "live", "live", "Arrival at A was back-calculated from departure, not observed"),
    ("hist_gain_mean_7d", "history", "history", "This train, this segment: mean delay gained, previous 7 days"),
    ("hist_gain_mean_30d", "history", "history", "Same, previous 30 days"),
    ("hist_gain_median_30d", "history", "history", "Same, median, previous 30 days"),
    ("hist_gain_std_30d", "history", "history", "Same, standard deviation, previous 30 days"),
    ("hist_gain_p90_30d", "history", "history", "Same, 90th percentile, previous 30 days"),
    ("hist_gain_mean_90d", "history", "history", "Same, mean, previous 90 days"),
    ("hist_n_30d", "history", "history", "Number of past runs behind the 30-day figures"),
    ("hist_arr_delay_mean_30d", "history", "history", "This train: mean arrival delay at B, previous 30 days"),
    ("hist_gain_last", "history", "history", "Delay gained on this segment on the train's previous run"),
    ("days_since_last_run", "history", "history", "Days since the train last ran this segment"),
    ("y_arr_delay_B", "target", "target", "TARGET: arrival delay at B (min)"),
    ("y_gain", "target", "target", "TARGET: delay gained A -> B = y_arr_delay_B - dep_delay_A (min)"),
    ("act_run_min", "target", "target", "Actual running time A -> B (min). Outcome, never an input"),
    ("has_target", "quality", "quality", "Both dep_delay_A and y_arr_delay_B exist"),
    ("arr_imputed_B", "quality", "quality", "Arrival at B was back-calculated from departure: target is approximate"),
    ("flag_mismatch_B", "quality", "quality", "Source arrival delay at B disagreed with the timestamps"),
    ("flag_speed", "quality", "quality", "Planned or actual speed above the limit, or zero running time"),
]
OUT_COLS = [c[0] for c in DICTIONARY]

GAIN_BINS = [-np.inf, -30.5, -15.5, -5.5, -0.5, 0.5, 5.5, 15.5, 30.5, 60.5, np.inf]
GAIN_LABELS = ["<= -31", "-30..-16", "-15..-6", "-5..-1", "0", "1..5", "6..15",
               "16..30", "31..60", "> 60"]

STATIONS = {}   # set in each worker by init_worker


def init_worker(stations):
    global STATIONS
    STATIONS = stations


def mins(x):
    return x.dt.total_seconds() / 60.0


def fl(s):
    return s.astype("float64")


HIST_COLS = ["hist_gain_mean_7d", "hist_gain_mean_30d", "hist_gain_median_30d",
             "hist_gain_std_30d", "hist_gain_p90_30d", "hist_gain_mean_90d", "hist_n_30d",
             "hist_arr_delay_mean_30d", "hist_gain_last", "days_since_last_run"]


def add_history(seg, windows):
    """
    Rolling stats per (train, from, to) using only runs on EARLIER dates.

    Works on a daily table with one row per (train, from, to, date): if a
    circular route covers the same pair twice in a day, that day's values are
    averaged. Results are joined back by key, never by row position, so the
    output does not depend on pandas' internal ordering. Missing keys are
    replaced by a placeholder so pandas cannot silently drop those rows.
    """
    keys = pd.DataFrame({k: seg[k].astype(object).where(seg[k].notna(), "<NA>").astype(str)
                         for k in SEG_KEY})
    keys["date"] = seg["date"].to_numpy()
    base = keys.assign(y_gain=seg["y_gain"].astype("float64").to_numpy(),
                       y_arr=seg["y_arr_delay_B"].astype("float64").to_numpy())

    daily = (base.groupby(SEG_KEY + ["date"], sort=True, dropna=False)
             .agg(y_gain=("y_gain", "mean"), y_arr=("y_arr", "mean"))
             .reset_index())
    di = daily.set_index("date")
    g = di.groupby(SEG_KEY, sort=False, dropna=False)

    def roll(col, w):
        # index of the result is (train, from, to, date) in every pandas version
        return g[col].rolling(f"{w}D", closed="left")

    w_short, w_mid, w_long = windows
    r30 = roll("y_gain", w_mid)
    stats = {
        "hist_gain_mean_7d": roll("y_gain", w_short).mean(),
        "hist_gain_mean_30d": r30.mean(),
        "hist_gain_median_30d": r30.median(),
        "hist_gain_std_30d": r30.std(),
        "hist_gain_p90_30d": r30.quantile(0.9),
        "hist_n_30d": r30.count(),
        "hist_gain_mean_90d": roll("y_gain", w_long).mean(),
        "hist_arr_delay_mean_30d": roll("y_arr", w_mid).mean(),
    }
    for name, ser in stats.items():
        ser = ser.copy()
        ser.index = ser.index.set_names(SEG_KEY + ["date"])
        daily = daily.join(ser.rename(name), on=SEG_KEY + ["date"])

    dg = daily.groupby(SEG_KEY, sort=False, dropna=False)
    daily["hist_gain_last"] = dg["y_gain"].shift(1)
    daily["days_since_last_run"] = (daily["date"] - dg["date"].shift(1)).dt.days
    daily["hist_n_30d"] = daily["hist_n_30d"].fillna(0).astype("int32")

    merged = base[SEG_KEY + ["date"]].merge(daily[SEG_KEY + ["date"] + HIST_COLS],
                                            on=SEG_KEY + ["date"], how="left", validate="many_to_one")
    if len(merged) != len(seg):
        raise RuntimeError("history merge changed the number of rows")
    out = seg.copy()
    for col in HIST_COLS:
        out[col] = merged[col].to_numpy()
    return out


def build_file(task):
    path, rel, out_root, cfg = task
    c, val = Counter(), Counter()
    out_path = os.path.join(out_root, rel)
    try:
        d = pd.read_parquet(path)
    except Exception as e:
        return c, val, f"{rel}: read failed: {type(e).__name__}: {str(e)[:100]}"
    c["files_ok"] += 1
    c["stop_rows"] += len(d)
    if d.empty:
        return c, val, None

    d = d.sort_values(JKEY + ["seq"]).reset_index(drop=True)
    jg = d.groupby(JKEY, sort=False)

    dep = fl(d["delay_dep_min"])
    arr = fl(d["delay_arr_min"])
    km = fl(d["distance_km"])
    sch_dwell = mins(d["sch_dep"] - d["sch_arr"])
    act_dwell = mins(d["act_dep"] - d["act_arr"])

    # per-stop helpers (everything here is known once the train has left that stop)
    st = pd.DataFrame(index=d.index)
    st["arr_imputed"] = (d["src_delay_arr_min"].isna() & (arr >= 2)).to_numpy()
    st["gain_into"] = arr - jg["delay_dep_min"].shift(1).astype("float64")
    st["dwell_overrun"] = act_dwell - sch_dwell
    st["origin_dep"] = dep.where(d["stop_idx"] == 0).groupby([d["date"], d["train_no"]]).transform("max")
    st["max_dep"] = dep.groupby([d["date"], d["train_no"]]).cummax()
    st["gain_prev2"] = st["gain_into"].groupby([d["date"], d["train_no"]]).shift(1)
    st["dep_back2"] = dep.groupby([d["date"], d["train_no"]]).shift(2)
    jmax_km = km.groupby([d["date"], d["train_no"]]).transform("max")

    def nxt(series):
        return series.groupby([d["date"], d["train_no"]]).shift(-1)

    a = d["stop_idx"] < (d["n_stops"] - 1)          # rows that start a segment
    if not a.any():
        c["files_no_segments"] += 1
        return c, val, None

    seg = pd.DataFrame({
        "date": d["date"],
        "train_no": d["train_no"],
        "train_type": d["train_type"],
        "from_stn": d["station_code"],
        "to_stn": nxt(d["station_code"]),
        "from_seq": d["seq"],
        "to_seq": nxt(d["seq"]),
        "seg_idx": d["stop_idx"],
        "n_segments": d["n_stops"] - 1,
        "km_A": km,
        "km_B": nxt(km),
        "jmax_km": jmax_km,
        "sch_arr_A": d["sch_arr"],
        "sch_dep_A": d["sch_dep"],
        "act_dep_A": d["act_dep"],
        "sch_arr_B": nxt(d["sch_arr"]),
        "act_arr_B": nxt(d["act_arr"]),
        "sch_dep_B": nxt(d["sch_dep"]),
        "dep_delay_A": dep,
        "arr_delay_A": arr,
        "dwell_overrun_A": st["dwell_overrun"],
        "gain_prev1": st["gain_into"],
        "gain_prev2": st["gain_prev2"],
        "trend_2stops": dep - st["dep_back2"],
        "origin_dep_delay": st["origin_dep"],
        "max_dep_delay_so_far": st["max_dep"],
        "arr_imputed_A": st["arr_imputed"],
        "y_arr_delay_B": nxt(arr),
        "arr_imputed_B": nxt(st["arr_imputed"].astype("float64")).fillna(0).astype(bool),
        "flag_mismatch_B": nxt(d["flag_delay_mismatch_arr"].astype("float64")).fillna(0).astype(bool),
    })[a.to_numpy()].reset_index(drop=True)
    c["segments_built"] += len(seg)

    # ---- schedule & geometry
    seg["sch_run_min"] = mins(seg["sch_arr_B"] - seg["sch_dep_A"])
    bad = seg["sch_run_min"].isna() | (seg["sch_run_min"] <= 0)
    c["drop_bad_schedule"] += int(bad.sum())
    seg = seg[~bad.to_numpy()].reset_index(drop=True)
    if seg.empty:
        c["files_no_segments"] += 1
        return c, val, None

    seg["seg_km"] = seg["km_B"] - seg["km_A"]
    seg["km_to_end"] = seg["jmax_km"] - seg["km_B"]
    seg["stations_passed"] = (seg["to_seq"] - seg["from_seq"] - 1).astype("Int64")
    seg["sch_speed_kmph"] = seg["seg_km"] / seg["sch_run_min"] * 60
    seg["sch_dwell_A_min"] = mins(seg["sch_dep_A"] - seg["sch_arr_A"])
    seg["sch_dwell_B_min"] = mins(seg["sch_dep_B"] - seg["sch_arr_B"])
    seg["frac_journey"] = seg["seg_idx"] / seg["n_segments"].clip(lower=1)
    seg["is_first_segment"] = (seg["seg_idx"] == 0).to_numpy()
    seg["is_last_segment"] = (seg["seg_idx"] == seg["n_segments"] - 1).to_numpy()
    seg["journey_day"] = (seg["sch_dep_A"].dt.normalize() - seg["date"]).dt.days.astype("Int64")

    # ---- calendar (from scheduled departure at A)
    t = seg["sch_dep_A"]
    seg["dep_hour"] = t.dt.hour + t.dt.minute / 60.0
    seg["hour_sin"] = np.sin(2 * np.pi * seg["dep_hour"] / 24)
    seg["hour_cos"] = np.cos(2 * np.pi * seg["dep_hour"] / 24)
    seg["dow"] = t.dt.dayofweek.astype("int8")
    seg["is_weekend"] = (seg["dow"] >= 5).to_numpy()
    seg["month"] = t.dt.month.astype("int8")
    seg["day_of_year"] = t.dt.dayofyear.astype("int16")

    # ---- stations
    for side, col in (("from", "from_stn"), ("to", "to_stn")):
        codes = seg[col].astype(object)
        seg[f"{side}_lat"] = codes.map(lambda k: STATIONS.get(k, (np.nan, np.nan))[0]).astype("float64")
        seg[f"{side}_lng"] = codes.map(lambda k: STATIONS.get(k, (np.nan, np.nan))[1]).astype("float64")
    seg["mid_lat"] = (seg["from_lat"] + seg["to_lat"]) / 2
    seg["mid_lng"] = (seg["from_lng"] + seg["to_lng"]) / 2
    c["missing_station_coords"] += int(seg[["from_lat", "to_lat"]].isna().any(axis=1).sum())
    n_na = int(seg[["from_stn", "to_stn"]].isna().any(axis=1).sum())
    if n_na:
        c["missing_station_code"] += n_na
        val[("na_file", rel)] += n_na

    # ---- targets
    seg["y_gain"] = seg["y_arr_delay_B"] - seg["dep_delay_A"]
    seg["act_run_min"] = mins(seg["act_arr_B"] - seg["act_dep_A"])
    seg["has_target"] = (seg["y_gain"].notna()).to_numpy()
    act_speed = seg["seg_km"] / seg["act_run_min"] * 60
    seg["flag_speed"] = ((seg["sch_speed_kmph"] > cfg["max_speed_kmph"])
                         | (act_speed > cfg["max_speed_kmph"])
                         | (seg["act_run_min"] <= 0)
                         | (seg["seg_km"] <= 0)).fillna(False).astype(bool)

    # ---- history (previous days only)
    seg = add_history(seg, cfg["hist_windows"])

    # ---- counts
    ht = seg["has_target"]
    c["segments_out"] += len(seg)
    c["segments_with_target"] += int(ht.sum())
    c["arr_imputed_B"] += int(seg["arr_imputed_B"].sum())
    c["arr_imputed_B_with_target"] += int((seg["arr_imputed_B"] & ht).sum())
    c["flag_mismatch_B"] += int(seg["flag_mismatch_B"].sum())
    c["flag_speed"] += int(seg["flag_speed"].sum())
    c["has_history_30d"] += int((seg["hist_n_30d"] > 0).sum())
    c["missing_dep_delay_A"] += int(seg["dep_delay_A"].isna().sum())
    c["missing_arr_delay_B"] += int(seg["y_arr_delay_B"].isna().sum())
    gains = seg.loc[ht.to_numpy(), "y_gain"]
    for k, n in pd.cut(gains, GAIN_BINS, labels=GAIN_LABELS).value_counts().items():
        val[("gain", "all", str(k))] += int(n)
    for tt, sub in seg[ht.to_numpy()].groupby("train_type"):
        val[("type_n", str(tt))] += len(sub)
        val[("type_gain_sum", str(tt))] += float(sub["y_gain"].sum())
        val[("type_gain_abs", str(tt))] += float(sub["y_gain"].abs().sum())
    for flag in ("arr_imputed_B",):
        sub = seg.loc[(ht & seg[flag]).to_numpy(), "y_gain"]
        for k, n in pd.cut(sub, GAIN_BINS, labels=GAIN_LABELS).value_counts().items():
            val[("gain", flag, str(k))] += int(n)

    seg = seg[OUT_COLS]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    seg.to_parquet(out_path, index=False)
    c["files_written"] += 1
    return c, val, None


def run_map(func, items, workers, stations):
    if workers <= 1:
        init_worker(stations)
        for it in items:
            yield func(it)
    else:
        with Pool(workers, initializer=init_worker, initargs=(stations,)) as pool:
            for r in pool.imap_unordered(func, items, chunksize=8):
                yield r


def main():
    ap = argparse.ArgumentParser(description="Build station-to-station segments and features.")
    ap.add_argument("path", nargs="?", default=DEFAULT_IN, help="cleaned data folder")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (trial run)")
    args = ap.parse_args()
    cfg = dict(CFG)

    in_dir, out_dir = os.path.abspath(args.path), os.path.abspath(args.out)
    if in_dir == out_dir or out_dir.startswith(in_dir + os.sep):
        sys.exit("Output folder must be outside the input folder.")
    st_path = os.path.join(in_dir, "_stations.parquet")
    if not os.path.exists(st_path):
        sys.exit(f"Station lookup not found: {st_path} (it is written by clean_train_data.py)")
    stn = pd.read_parquet(st_path)
    stations = {str(r.station_code): (float(r.stn_lat), float(r.stn_lng))
                for r in stn.itertuples(index=False)}

    files = [f for f in sorted(glob.glob(os.path.join(in_dir, "**", "*.parquet"), recursive=True))
             if not os.path.basename(f).startswith("_")]
    if not files:
        sys.exit(f"No cleaned .parquet files found under {in_dir}")
    if args.limit:
        files = files[:args.limit]
    os.makedirs(out_dir, exist_ok=True)
    print(f"Files: {len(files):,} | stations: {len(stations):,} | workers: {args.workers} | output: {out_dir}")

    t0 = time.time()
    tasks = [(f, os.path.relpath(f, in_dir), out_dir, cfg) for f in files]
    C, V, errors = Counter(), Counter(), []
    for i, (c, val, err) in enumerate(run_map(build_file, tasks, args.workers, stations), 1):
        C.update(c)
        V.update(val)
        if err:
            errors.append(err)
        if i % 500 == 0 or i == len(tasks):
            print(f"  [{i:>6,}/{len(tasks):,}] segments {C['segments_out']:>12,}  ({time.time() - t0:,.0f}s)")

    # ---- data dictionary
    pd.DataFrame(DICTIONARY, columns=["column", "group", "known_when", "description"]).to_csv(
        os.path.join(out_dir, "_data_dictionary.csv"), index=False)

    # ---- log
    L = []
    out = lambda *a: L.append(" ".join(str(x) for x in a))
    seg_n = max(C["segments_out"], 1)
    tgt_n = max(C["segments_with_target"], 1)
    out("=" * 78)
    out("SEGMENT BUILD LOG")
    out("=" * 78)
    out(f"Input : {in_dir}")
    out(f"Output: {out_dir}")
    out(f"Files : {len(files):,} processed | {C['files_written']:,} written | "
        f"{len(errors)} failed | {C['files_no_segments']} with no segments")
    out(f"Time  : {time.time() - t0:,.0f}s")
    out("")
    out("-" * 78)
    out("COUNTS")
    out("-" * 78)
    rows = [
        ("Stop rows read", C["stop_rows"], None),
        ("Segments built (consecutive stops)", C["segments_built"], None),
        ("Dropped: scheduled run time missing or <= 0", C["drop_bad_schedule"], C["segments_built"]),
        ("Segments written", C["segments_out"], None),
        ("   with a target (dep at A and arr at B both known)", C["segments_with_target"], seg_n),
        ("   missing departure delay at A", C["missing_dep_delay_A"], seg_n),
        ("   missing arrival delay at B", C["missing_arr_delay_B"], seg_n),
        ("   arrival at B back-calculated (approximate target)", C["arr_imputed_B_with_target"], tgt_n),
        ("   flagged: source delay mismatch at B", C["flag_mismatch_B"], seg_n),
        ("   flagged: speed above limit / zero run time / zero distance", C["flag_speed"], seg_n),
        ("   with 30-day history for this train + segment", C["has_history_30d"], seg_n),
        ("   missing station coordinates", C["missing_station_coords"], seg_n),
        ("   missing station code at A or B", C["missing_station_code"], seg_n),
    ]
    for label, n, base in rows:
        pct = f"{100 * n / max(base, 1):6.2f}%" if base else "       "
        out(f"{n:>14,}  {pct}  {label}")
    out("")
    out("Percentages under 'with a target' / 'back-calculated' use segments-with-target as the base;")
    out("the rest use segments written.")
    na_files = sorted(((n, k[1]) for k, n in V.items() if k[0] == "na_file"), reverse=True)
    if na_files:
        out("")
        out(f"Files with missing station codes ({len(na_files)}), most affected first:")
        for n, f in na_files[:15]:
            out(f"   {f}: {n:,} segments")

    out("")
    out("-" * 78)
    out("TARGET: DELAY GAINED A -> B (minutes), segments with a target")
    out("-" * 78)
    tot_all = sum(V[("gain", "all", b)] for b in GAIN_LABELS)
    tot_imp = sum(V[("gain", "arr_imputed_B", b)] for b in GAIN_LABELS)
    out(f"{'bucket':>10} {'all %':>8} {'imputed-B %':>12}")
    for b in GAIN_LABELS:
        pa = 100 * V[("gain", "all", b)] / max(tot_all, 1)
        pi = 100 * V[("gain", "arr_imputed_B", b)] / max(tot_imp, 1)
        out(f"{b:>10} {pa:>8.1f} {pi:>12.1f}")

    out("")
    out("-" * 78)
    out("TARGET BY TRAIN TYPE (segments with a target)")
    out("-" * 78)
    types = sorted({k[1] for k in V if k[0] == "type_n"}, key=lambda t: -V[("type_n", t)])
    out(f"{'type':<16} {'segments':>12} {'mean gain':>10} {'mean |gain|':>12}")
    for tt in types[:20]:
        n = V[("type_n", tt)]
        out(f"{tt:<16} {n:>12,} {V[('type_gain_sum', tt)] / n:>10.2f} {V[('type_gain_abs', tt)] / n:>12.2f}")

    out("")
    out("-" * 78)
    out("COLUMNS (full descriptions in _data_dictionary.csv)")
    out("-" * 78)
    for kw in ("schedule", "live", "history", "target", "quality"):
        cols = [c[0] for c in DICTIONARY if c[2] == kw]
        out(f"{kw:<9} ({len(cols):>2}): {', '.join(cols)}")

    if errors:
        out("")
        out(f"ERRORS ({len(errors)})")
        L.extend(errors[:30])

    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "_segments_log.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved log to {os.path.join(out_dir, '_segments_log.txt')}")


if __name__ == "__main__":
    freeze_support()
    main()
