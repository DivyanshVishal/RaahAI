#!/usr/bin/env python3
"""
Network features (stage 2 of feature engineering)
-------------------------------------------------
Adds features that need ALL trains at once to the segment files written by
build_segments.py, and writes the final feature table.

    congestion   other trains scheduled at A / at B / on the same section
                 around the same time; opposing trains on the section;
                 higher-priority trains just ahead
    busyness     trains departing station A that day
    history      all trains on this section / at these stations over the
                 PREVIOUS 7 or 30 days (never the same day)
    live input   known_delay_A: departure delay at A, or arrival delay at A
                 when the departure was not recorded
    holidays     optional, from a CSV you supply (--holidays)

Only trains present in this dataset are counted (no freight, no untracked
trains), so congestion counts are relative measures, not absolute traffic.

Memory: keeps ~40 bytes per segment in compact integer arrays (~1 GB for
26M segments), then rewrites each train file with the new columns.

Usage:
    python add_network_features.py
    python add_network_features.py processed/segments/year=2026 --out processed/features/year=2026
    python add_network_features.py --holidays holidays_2026.csv
    python add_network_features.py --limit 300        # quick trial (counts cover only those trains)

Outputs (in --out):
    <same relative paths>.parquet   segment rows with all features
    _features_log.txt               counts and feature summaries
    _data_dictionary.csv            every column, its group, and when it is known

Requires: pandas, numpy, pyarrow
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

DEFAULT_IN = "processed/segments/year=2026"
DEFAULT_OUT = "processed/features/year=2026"
EPOCH = pd.Timestamp("2025-12-01")   # times are stored as minutes since this

# Remaining duplicate train-type labels seen in the segment log. Edit if wrong.
TYPE_MAP = {"EMU - MUMBAI": "EMU", "EMU - DELHI": "EMU", "EMU - KOLKATA": "EMU",
            "PASSENGER": "PAS", "MAIL/EXPRESS": "MEX", "SUPERFAST": "SUF"}

# ASSUMED priority ranks (higher = gets precedence on a shared line). Used only
# for the "higher-priority train ahead" count; train_type itself stays a feature.
# Please check against your railway's actual precedence rules and edit.
PRIORITY = {
    "RAJ": 5, "SHT": 5, "DRNT": 5,
    "SUF": 4, "JSH": 4, "GBR": 4, "AC SUPERFAST": 4, "AC EXPRESS": 4, "Humsafar": 4,
    "MEX": 3, "ANT": 3, "TOD": 3,
    "PAS": 2, "MEMU": 2, "DEMU": 2, "EMU": 2, "SUB": 2,
}
DEFAULT_PRIORITY = 3

LIGHT_COLS = ["date", "train_no", "train_type", "from_stn", "to_stn", "sch_dep_A",
              "sch_arr_B", "y_gain", "y_arr_delay_B", "dep_delay_A", "flag_speed"]

NEW_DICT = [
    ("known_delay_A", "live", "live", "Latest known delay at A: departure delay, or arrival delay if departure not recorded (min)"),
    ("known_delay_A_src", "live", "live", "Where known_delay_A came from: dep / arr / none"),
    ("prio_rank", "train", "schedule", "ASSUMED priority rank of this train type (5 = highest); see PRIORITY in the script"),
    ("stn_A_deps_30m", "congestion", "schedule", "Other trains scheduled to depart A within +/-30 min"),
    ("stn_B_arrs_30m", "congestion", "schedule", "Other trains scheduled to arrive at B within +/-30 min"),
    ("sec_same_dir_60m", "congestion", "schedule", "Other trains scheduled on the same A->B section, departing within +/-60 min"),
    ("sec_ahead_30m", "congestion", "schedule", "Trains scheduled on A->B departing in the 30 min before this one"),
    ("sec_higher_prio_ahead_30m", "congestion", "schedule", "Of those, trains with a higher priority rank"),
    ("sec_opposing", "congestion", "schedule", "Trains scheduled B->A that leave B between 30 min before this train leaves A and its scheduled arrival at B"),
    ("stn_A_daily_deps", "busyness", "schedule", "Trains scheduled to depart A on that calendar day"),
    ("sec_all_gain_mean_7d", "history", "history", "All trains on A->B: mean delay gained, previous 7 days"),
    ("sec_all_gain_mean_30d", "history", "history", "All trains on A->B: mean delay gained, previous 30 days"),
    ("sec_all_gain_std_30d", "history", "history", "All trains on A->B: std of delay gained, previous 30 days"),
    ("sec_all_n_30d", "history", "history", "Runs behind the 30-day section figures"),
    ("stn_A_dep_delay_mean_30d", "history", "history", "All trains at A: mean departure delay, previous 30 days"),
    ("stn_B_arr_delay_mean_30d", "history", "history", "All trains at B: mean arrival delay, previous 30 days"),
    ("is_holiday", "calendar", "schedule", "Departure day is in the holiday file (only with --holidays)"),
    ("days_to_holiday", "calendar", "schedule", "Days until the next holiday, capped at 15 (only with --holidays)"),
    ("days_since_holiday", "calendar", "schedule", "Days since the last holiday, capped at 15 (only with --holidays)"),
]


# ------------------------------------------------------------------ helpers
class Codes:
    """Maps strings to compact integer codes as files are read."""
    def __init__(self):
        self.map = {}

    def encode(self, s):
        vals = s.astype(object).where(s.notna(), "<NA>").to_numpy()
        new = [v for v in pd.unique(vals) if v not in self.map]
        for v in new:
            self.map[v] = len(self.map)
        return pd.Series(vals).map(self.map).to_numpy(np.int32)


def to_minutes(ts):
    m = (ts - EPOCH).dt.total_seconds() / 60.0
    return np.round(m.to_numpy(np.float64)).astype(np.float64)   # NaN kept


def count_window(ref_g, ref_t, q_g, q_t, lo, hi):
    """
    For each query, number of reference events in the same group with time in
    [q_t + lo, q_t + hi]. lo / hi may be scalars or arrays (minutes).
    """
    M = np.int64(10_000_000)
    ok = ~np.isnan(ref_t)
    ref = np.sort(ref_g[ok].astype(np.int64) * M + ref_t[ok].astype(np.int64))
    out = np.zeros(len(q_g), dtype=np.int32)
    qok = ~np.isnan(q_t)
    if not qok.any() or not len(ref):
        return out
    base = q_g[qok].astype(np.int64) * M + q_t[qok].astype(np.int64)
    lo_a = np.broadcast_to(np.asarray(lo, dtype=np.float64), q_t.shape)[qok]
    hi_a = np.broadcast_to(np.asarray(hi, dtype=np.float64), q_t.shape)[qok]
    lo_k = base + np.ceil(lo_a).astype(np.int64)
    hi_k = base + np.floor(hi_a).astype(np.int64)
    n = np.searchsorted(ref, hi_k, "right") - np.searchsorted(ref, lo_k, "left")
    out[qok] = np.maximum(n, 0)
    return out


def prev_days_stats(group, day, val, window):
    """
    For each row: mean, std and count of `val` over rows in the same group whose
    day is in [day - window, day - 1]. The row's own day is always excluded.
    """
    K = np.int64(100_000)
    ok = ~np.isnan(val)
    key = group[ok].astype(np.int64) * K + day[ok].astype(np.int64)
    v = val[ok].astype(np.float64)
    order = np.argsort(key, kind="stable")
    key, v = key[order], v[order]
    del order
    # one entry per (group, day): sums via boundaries of the sorted keys
    if len(key):
        start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        keys = key[start]
        s_day = np.add.reduceat(v, start)
        ss_day = np.add.reduceat(v * v, start)
        n_day = np.diff(np.r_[start, len(key)])
    else:
        keys = np.zeros(0, np.int64)
        s_day = ss_day = np.zeros(0)
        n_day = np.zeros(0, np.int64)
    del key, v
    cs = np.r_[0.0, np.cumsum(s_day)]
    css = np.r_[0.0, np.cumsum(ss_day)]
    cn = np.r_[0, np.cumsum(n_day)]
    del s_day, ss_day, n_day
    q = group.astype(np.int64) * K + day.astype(np.int64)
    lo = np.searchsorted(keys, q - window, "left")
    hi = np.searchsorted(keys, q, "left")              # excludes the row's own day
    del q
    n = (cn[hi] - cn[lo]).astype(np.float64)
    s = cs[hi] - cs[lo]
    ss = css[hi] - css[lo]
    del lo, hi
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
        var = np.where(n > 1, (ss - s * s / np.where(n > 0, n, 1)) / np.where(n > 1, n - 1, 1), np.nan)
    std = np.sqrt(np.clip(var, 0, None))
    return mean.astype(np.float32), std.astype(np.float32), n.astype(np.int32)


def read_light(path):
    try:
        return pd.read_parquet(path, columns=LIGHT_COLS), None
    except Exception as e:
        return None, f"{os.path.basename(path)}: {type(e).__name__}: {str(e)[:100]}"


def summarize(name, x):
    x = pd.Series(x).astype("float64").dropna()
    if not len(x):
        return f"{name:<28} (empty)"
    q = x.quantile([0.5, 0.9, 0.99])
    return (f"{name:<28} mean {x.mean():8.2f} | median {q[0.5]:7.1f} | p90 {q[0.9]:7.1f} | "
            f"p99 {q[0.99]:7.1f} | max {x.max():8.1f}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Add cross-train network features.")
    ap.add_argument("path", nargs="?", default=DEFAULT_IN, help="segments folder from build_segments.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=max(2, min(8, os.cpu_count() or 2)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--holidays", default="", help="optional CSV with a 'date' column (YYYY-MM-DD)")
    args = ap.parse_args()

    in_dir, out_dir = os.path.abspath(args.path), os.path.abspath(args.out)
    if in_dir == out_dir or out_dir.startswith(in_dir + os.sep):
        sys.exit("Output folder must be outside the input folder.")
    files = [f for f in sorted(glob.glob(os.path.join(in_dir, "**", "*.parquet"), recursive=True))
             if not os.path.basename(f).startswith("_")]
    if not files:
        sys.exit(f"No segment files found under {in_dir}")
    if args.limit:
        files = files[:args.limit]
    holidays = None
    if args.holidays:
        h = pd.read_csv(args.holidays)
        if "date" not in h.columns:
            sys.exit("Holiday CSV needs a 'date' column")
        holidays = np.sort(pd.to_datetime(h["date"]).dt.normalize().unique())
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"Files: {len(files):,} | threads: {args.threads} | output: {out_dir}")

    # ---------------- pass 1: read compact columns from every file
    print("Pass 1: reading schedule and delay columns ...")
    stn, trn = Codes(), Codes()
    parts, offsets, errors = [], [0], []
    with ThreadPoolExecutor(args.threads) as ex:
        for i, (df, err) in enumerate(ex.map(read_light, files), 1):
            if err:
                errors.append(err)
                df = pd.DataFrame(columns=LIGHT_COLS)
            ttype = df["train_type"].astype(object).replace(TYPE_MAP)
            parts.append({
                "from": stn.encode(df["from_stn"]),
                "to": stn.encode(df["to_stn"]),
                "train": trn.encode(df["train_no"]),
                "prio": ttype.map(PRIORITY).fillna(DEFAULT_PRIORITY).to_numpy(np.int8),
                "dep": to_minutes(df["sch_dep_A"]),
                "arr": to_minutes(df["sch_arr_B"]),
                "gain": df["y_gain"].astype("float64").to_numpy(np.float32),
                "yarr": df["y_arr_delay_B"].astype("float64").to_numpy(np.float32),
                "depdelay": df["dep_delay_A"].astype("float64").to_numpy(np.float32),
                "flag": df["flag_speed"].astype(bool).to_numpy(),
            })
            offsets.append(offsets[-1] + len(df))
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i:>6,}/{len(files):,}] segments {offsets[-1]:>12,}  ({time.time() - t0:,.0f}s)")
    A = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    n = len(A["from"])
    S = np.int64(len(stn.map) + 1)
    pair = A["from"].astype(np.int64) * S + A["to"]
    rpair = A["to"].astype(np.int64) * S + A["from"]
    day = np.floor(A["dep"] / 1440.0)                       # calendar day of departure from A
    print(f"  {n:,} segments | {len(stn.map):,} stations | {len(trn.map):,} trains | "
          f"arrays ~{sum(v.nbytes for v in A.values()) / 1e9:.2f} GB")

    # ---------------- congestion (schedule-based)
    print("Computing congestion and history features ...")
    F = {}
    F["stn_A_deps_30m"] = count_window(A["from"], A["dep"], A["from"], A["dep"], -30, 30) - 1
    F["stn_B_arrs_30m"] = count_window(A["to"], A["arr"], A["to"], A["arr"], -30, 30) - 1
    F["sec_same_dir_60m"] = count_window(pair, A["dep"], pair, A["dep"], -60, 60) - 1
    F["sec_ahead_30m"] = count_window(pair, A["dep"], pair, A["dep"], -30, -1)
    higher = np.zeros(n, dtype=np.int32)
    for p in np.unique(A["prio"]):
        sel = A["prio"] == p
        cnt = count_window(pair[sel], A["dep"][sel], pair, A["dep"], -30, -1)
        higher += np.where(A["prio"] < p, cnt, 0).astype(np.int32)
    F["sec_higher_prio_ahead_30m"] = higher
    run = A["arr"] - A["dep"]
    F["sec_opposing"] = count_window(pair, A["dep"], rpair, A["dep"], -30, run)
    # scheduled times are always present (build_segments drops segments without them),
    # so "-1" only removes the train itself
    for k in ("stn_A_deps_30m", "stn_B_arrs_30m", "sec_same_dir_60m"):
        F[k] = np.maximum(F[k], 0).astype(np.int32)
    dkey = A["from"].astype(np.int64) * np.int64(100_000) + day.astype(np.int64)
    _, inv, cnt = np.unique(dkey, return_inverse=True, return_counts=True)
    F["stn_A_daily_deps"] = cnt[inv.ravel()].astype(np.int32)
    del dkey, inv, cnt

    # ---------------- all-train history (previous days only)
    usable = (~A["flag"]).astype(bool)
    gain_h = np.where(usable, A["gain"], np.nan)
    m7, _, _ = prev_days_stats(pair, day, gain_h, 7)
    m30, s30, n30 = prev_days_stats(pair, day, gain_h, 30)
    F["sec_all_gain_mean_7d"], F["sec_all_gain_mean_30d"] = m7, m30
    F["sec_all_gain_std_30d"], F["sec_all_n_30d"] = s30, n30
    F["stn_A_dep_delay_mean_30d"], _, _ = prev_days_stats(A["from"], day, A["depdelay"], 30)
    arrday = np.floor(A["arr"] / 1440.0)
    F["stn_B_arr_delay_mean_30d"], _, _ = prev_days_stats(A["to"], arrday, A["yarr"], 30)

    # ---------------- holidays (optional)
    if holidays is not None:
        hd = ((holidays - np.datetime64(EPOCH)) / np.timedelta64(1, "D")).astype(np.int64)
        dv = np.nan_to_num(day, nan=-10 ** 6).astype(np.int64)
        idx = np.searchsorted(hd, dv, "left")
        nxt = np.where(idx < len(hd), hd[np.minimum(idx, len(hd) - 1)] - dv, 10 ** 6)
        prv_idx = np.searchsorted(hd, dv, "right") - 1
        prv = np.where(prv_idx >= 0, dv - hd[np.maximum(prv_idx, 0)], 10 ** 6)
        F["is_holiday"] = (nxt == 0)
        F["days_to_holiday"] = np.minimum(nxt, 15).astype(np.int16)
        F["days_since_holiday"] = np.minimum(prv, 15).astype(np.int16)
    F["prio_rank"] = A["prio"]
    del pair, rpair, run, day, arrday, gain_h, usable
    for k in ("from", "to", "train", "dep", "arr", "flag"):
        del A[k]
    print(f"  done ({time.time() - t0:,.0f}s)")

    # ---------------- pass 2: write files with the new columns
    print("Pass 2: writing feature files ...")
    feat_names = list(F.keys())

    def write_one(i):
        src = files[i]
        lo, hi = offsets[i], offsets[i + 1]
        try:
            df = pd.read_parquet(src)
        except Exception as e:
            return 0, f"{os.path.basename(src)}: {type(e).__name__}: {str(e)[:100]}"
        if len(df) != hi - lo:
            return 0, f"{os.path.basename(src)}: row count changed since pass 1"
        df["train_type"] = df["train_type"].astype(object).replace(TYPE_MAP).astype("string")
        dep, arr = df["dep_delay_A"].astype("float64"), df["arr_delay_A"].astype("float64")
        df["known_delay_A"] = dep.fillna(arr)
        df["known_delay_A_src"] = np.where(dep.notna(), "dep", np.where(arr.notna(), "arr", "none"))
        for k in feat_names:
            df[k] = F[k][lo:hi]
        out_path = os.path.join(out_dir, os.path.relpath(src, in_dir))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_parquet(out_path, index=False)
        return len(df), None

    written = 0
    with ThreadPoolExecutor(args.threads) as ex:
        for i, (nrows, err) in enumerate(ex.map(write_one, range(len(files))), 1):
            written += nrows
            if err:
                errors.append(err)
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i:>6,}/{len(files):,}] rows written {written:>12,}  ({time.time() - t0:,.0f}s)")

    # ---------------- dictionary
    dict_in = os.path.join(in_dir, "_data_dictionary.csv")
    base = pd.read_csv(dict_in) if os.path.exists(dict_in) else pd.DataFrame(
        columns=["column", "group", "known_when", "description"])
    new = pd.DataFrame([r for r in NEW_DICT if r[0] in feat_names or r[0].startswith("known_delay")],
                       columns=["column", "group", "known_when", "description"])
    pd.concat([base, new], ignore_index=True).to_csv(os.path.join(out_dir, "_data_dictionary.csv"), index=False)

    # ---------------- log
    L = []
    out = lambda *a: L.append(" ".join(str(x) for x in a))
    out("=" * 78)
    out("NETWORK FEATURES LOG")
    out("=" * 78)
    out(f"Input : {in_dir}")
    out(f"Output: {out_dir}")
    out(f"Files : {len(files):,} | rows written {written:,} of {n:,} | errors {len(errors)}")
    out(f"Time  : {time.time() - t0:,.0f}s")
    if args.limit:
        out("!! --limit was used: congestion and section history only count the trains in this subset.")
    out("")
    out("-" * 78)
    out("LIVE INPUT: known_delay_A")
    out("-" * 78)
    dep_ok = ~np.isnan(A["depdelay"])
    out(f"departure delay at A recorded          {int(dep_ok.sum()):>12,}  ({100 * dep_ok.mean():5.1f}%)")
    out("Rows without it use the arrival delay at A instead (see known_delay_A_src).")
    yb = ~np.isnan(A["yarr"])
    out(f"target y_arr_delay_B available         {int(yb.sum()):>12,}  ({100 * yb.mean():5.1f}%)")
    out(f"target y_gain available                {int((~np.isnan(A['gain'])).sum()):>12,}  "
        f"({100 * (~np.isnan(A['gain'])).mean():5.1f}%)")
    out("")
    out("-" * 78)
    out("FEATURE SUMMARIES")
    out("-" * 78)
    for k in feat_names:
        if k in ("is_holiday",):
            out(f"{k:<28} share True {100 * np.mean(F[k]):.2f}%")
        else:
            out(summarize(k, F[k]))
    out("")
    out("Priority ranks used (ASSUMED, edit PRIORITY in the script if wrong):")
    for p in sorted(set(PRIORITY.values()), reverse=True):
        out(f"  {p}: " + ", ".join(sorted(t for t, v in PRIORITY.items() if v == p)))
    out(f"  {DEFAULT_PRIORITY}: any other type (default)")
    out("Train-type merges: " + ", ".join(f"{a} -> {b}" for a, b in TYPE_MAP.items()))
    if holidays is None:
        out("Holidays: not added (no --holidays file given)")
    else:
        out(f"Holidays: {len(holidays)} dates from {args.holidays}")
    if errors:
        out("")
        out(f"ERRORS ({len(errors)})")
        L.extend(errors[:30])
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "_features_log.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved log to {os.path.join(out_dir, '_features_log.txt')}")


if __name__ == "__main__":
    main()
