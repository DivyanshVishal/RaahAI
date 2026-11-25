#!/usr/bin/env python3
"""
Multi-stop delay models with prediction ranges
----------------------------------------------
Extends the baseline from "next station" to ANY later station, and gives every
prediction a range (10th to 90th percentile) as well as a central value.

    LIVE (multi-stop)   The train has just left (or reached) station A, so its
                        current delay is known. Predicts the arrival delay at any
                        later station B on the same journey, 1 to n stops ahead.
                        Trained on (A, B) pairs sampled from each journey, with
                        features summed along the path between them.
    PRE-DEPARTURE       Nothing about today's run is known. Predicts the arrival
                        delay at each station from timetable, network and history
                        features (works for any station already).

Each gets three LightGBM models: p50 (central estimate) and p10 / p90 (range).
About 80% of actual delays should fall inside the p10-p90 range; the report checks.

Split (same as the baseline, edit with --val-start / --test-start):
    train before July, validation July (early stopping), test August.

Usage:
    python train_multistop.py --quick      # small trial
    python train_multistop.py              # full run

After training, use predict.py (same folder) to get predictions for a specific train.

Outputs (in --out, default models/multistop):
    live_p10.txt live_p50.txt live_p90.txt pre_p10.txt pre_p50.txt pre_p90.txt
    _multistop_report.txt       accuracy by stops ahead / time ahead, range coverage
    feature_lists.json          exact features of every model
    test_predictions.parquet    sample of test predictions for error analysis

Requires: pandas, numpy, pyarrow, lightgbm
"""
import argparse
import datetime
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("LightGBM is not installed. Run:  pip install lightgbm")

DEFAULT_IN = "processed/features/year=2026"
DEFAULT_OUT = "models/multistop"
CATEGORICAL = ["train_type", "known_delay_A_src"]
EXCLUDE = {"date", "train_no", "from_stn", "to_stn", "from_seq", "to_seq",
           "sch_dep_A", "sch_arr_B", "month", "day_of_year"}

# columns read from each feature file
READ = ["date", "train_no", "train_type", "seg_idx", "n_segments", "from_stn", "to_stn",
        "sch_dep_A", "sch_arr_B", "seg_km", "sch_dwell_A_min", "stations_passed",
        "known_delay_A", "known_delay_A_src", "dep_delay_A", "arr_delay_A", "dwell_overrun_A",
        "gain_prev1", "gain_prev2", "trend_2stops", "origin_dep_delay", "max_dep_delay_so_far",
        "arr_imputed_A", "dep_hour", "hour_sin", "hour_cos", "dow", "frac_journey",
        "from_lat", "from_lng", "to_lat", "to_lng", "km_to_end",
        "hist_gain_median_30d", "hist_gain_mean_90d", "hist_gain_p90_30d", "sec_all_gain_mean_30d",
        "sec_same_dir_60m", "sec_opposing", "stn_A_dep_delay_mean_30d", "stn_A_daily_deps",
        "stn_B_arr_delay_mean_30d", "hist_arr_delay_mean_30d", "is_last_segment",
        "y_arr_delay_B", "flag_speed", "arr_imputed_B"]

LIVE_FEATS = [
    # where the train is now (origin of the pair) - all known when it leaves A
    "known_delay_A", "known_delay_A_src", "dep_delay_A", "arr_delay_A", "dwell_overrun_A",
    "gain_prev1", "gain_prev2", "trend_2stops", "origin_dep_delay", "max_dep_delay_so_far",
    "arr_imputed_A", "train_type", "dep_hour", "hour_sin", "hour_cos", "dow", "seg_idx",
    "n_segments", "frac_A", "from_lat", "from_lng", "stn_A_dep_delay_mean_30d", "stn_A_daily_deps",
    # the path from A to the target station (timetable + history only)
    "stops_ahead", "path_sch_min", "path_km", "path_dwell_sch_min", "path_stations_passed",
    "path_hist_median_sum", "path_hist_mean90_sum", "path_hist_p90_sum", "path_sec_gain_sum",
    "path_hist_missing", "path_same_dir_sum", "path_opposing_sum",
    # the target station
    "to_lat", "to_lng", "frac_B", "km_to_end_B", "is_last_B", "stn_B_arr_delay_mean_30d",
    "hist_arr_delay_mean_30d",
]

PARAMS = {"learning_rate": 0.1, "num_leaves": 255, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l2": 1.0, "max_bin": 255, "verbose": -1}
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

# Combining live and pre-departure predictions: one weight per cell of
# (stops ahead) x (current delay at A), fitted on the validation month.
BLEND_STOPS_EDGES = [1, 3, 6, 10, 20]     # cells: 1 | 2-3 | 4-6 | 7-10 | 11-20 | 21+
BLEND_STOPS_LABELS = ["1", "2-3", "4-6", "7-10", "11-20", "21+"]
BLEND_DELAY_EDGES = [15, 60]              # cells: <=15 | 16-60 | >60 minutes
BLEND_DELAY_LABELS = ["<=15", "16-60", ">60"]
BLEND_MIN_ROWS = 500


def blend_cells(stops_ahead, known_delay):
    si = np.searchsorted(BLEND_STOPS_EDGES, np.asarray(stops_ahead, dtype="float64"), side="left")
    ki = np.searchsorted(BLEND_DELAY_EDGES, np.asarray(known_delay, dtype="float64"), side="left")
    return si, ki


def apply_blend(P_live, P_pre, stops_ahead, known_delay, weights, scale):
    """Weighted mix of live and pre-departure quantiles, then the calibrated range.
    P_* are (n, 3) arrays of p10, p50, p90. Returns best estimate, low, high."""
    si, ki = blend_cells(stops_ahead, known_delay)
    w = np.asarray(weights, dtype="float64")[si, ki][:, None]
    P = w * P_live + (1 - w) * P_pre
    P.sort(axis=1)
    return P[:, 1], P[:, 1] - scale * (P[:, 1] - P[:, 0]), P[:, 1] + scale * (P[:, 2] - P[:, 1])


# --------------------------------------------------------------- data build
def f32(s):
    return pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()


def make_pairs(df, ai, bj, st, ln):
    """
    Feature rows for (A, B) pairs on the same journey. `df` is one train's feature
    rows sorted by date, train_no, seg_idx. ai = row of the segment leaving A,
    bj = row of the segment arriving at B (bj >= ai), st / ln = first row and number
    of rows of that journey. Shared by training and predict.py so both build
    exactly the same inputs.
    """
    ai, bj, st, ln = (np.asarray(x, dtype=np.int64) for x in (ai, bj, st, ln))

    def csum(col, fill=0.0):
        v = np.nan_to_num(f32(df[col]), nan=fill)
        return np.cumsum(v), v

    out = {}
    for c in ["known_delay_A", "dep_delay_A", "arr_delay_A", "dwell_overrun_A", "gain_prev1",
              "gain_prev2", "trend_2stops", "origin_dep_delay", "max_dep_delay_so_far",
              "dep_hour", "hour_sin", "hour_cos", "dow", "seg_idx", "n_segments", "from_lat",
              "from_lng", "stn_A_dep_delay_mean_30d", "stn_A_daily_deps"]:
        out[c] = f32(df[c])[ai]
    out["arr_imputed_A"] = df["arr_imputed_A"].astype(bool).to_numpy()[ai].astype("float64")
    out["known_delay_A_src"] = df["known_delay_A_src"].astype(object).to_numpy()[ai]
    out["train_type"] = df["train_type"].astype(object).to_numpy()[ai]
    out["frac_A"] = f32(df["frac_journey"])[ai]

    out["stops_ahead"] = (bj - ai + 1).astype("float64")
    sdep = (df["sch_dep_A"] - pd.Timestamp("2025-12-01")).dt.total_seconds().to_numpy() / 60
    sarr = (df["sch_arr_B"] - pd.Timestamp("2025-12-01")).dt.total_seconds().to_numpy() / 60
    out["path_sch_min"] = sarr[bj] - sdep[ai]
    for name, col in (("path_km", "seg_km"), ("path_stations_passed", "stations_passed"),
                      ("path_hist_mean90_sum", "hist_gain_mean_90d"),
                      ("path_hist_p90_sum", "hist_gain_p90_30d"),
                      ("path_sec_gain_sum", "sec_all_gain_mean_30d"),
                      ("path_same_dir_sum", "sec_same_dir_60m"), ("path_opposing_sum", "sec_opposing")):
        cs, v = csum(col)
        out[name] = cs[bj] - cs[ai] + v[ai]
    # usual gain per segment: this train's median, else all trains' mean, else 0
    hm = f32(df["hist_gain_median_30d"])
    usual = np.where(np.isnan(hm), np.nan_to_num(f32(df["sec_all_gain_mean_30d"]), nan=0.0), hm)
    cs = np.cumsum(usual)
    out["path_hist_median_sum"] = cs[bj] - cs[ai] + usual[ai]
    miss = np.isnan(hm).astype("float64")
    cs = np.cumsum(miss)
    out["path_hist_missing"] = cs[bj] - cs[ai] + miss[ai]
    cs, _ = csum("sch_dwell_A_min")                  # stops strictly between A and B
    out["path_dwell_sch_min"] = cs[bj] - cs[ai]

    for c in ["to_lat", "to_lng", "stn_B_arr_delay_mean_30d", "hist_arr_delay_mean_30d"]:
        out[c] = f32(df[c])[bj]
    out["frac_B"] = ((bj - st + 1) / np.maximum(ln, 1)).astype("float64")
    out["km_to_end_B"] = f32(df["km_to_end"])[bj]
    out["is_last_B"] = df["is_last_segment"].astype(bool).to_numpy()[bj].astype("float64")

    out["date"] = df["date"].to_numpy()[ai]
    out["train_no"] = df["train_no"].astype(object).to_numpy()[ai]
    out["from_stn"] = df["from_stn"].astype(object).to_numpy()[ai]
    out["to_stn"] = df["to_stn"].astype(object).to_numpy()[bj]
    out["y_arr_delay_B"] = f32(df["y_arr_delay_B"])[bj]
    out["arr_imputed_B"] = df["arr_imputed_B"].astype(bool).to_numpy()[bj]
    pairs = pd.DataFrame(out)
    for c in pairs.columns:
        if pairs[c].dtype == "float64":
            pairs[c] = pairs[c].astype("float32")
    return pairs


def build_file(df, P, q_of_date, p_pre_of_date, rng, v0, t1, pre_feats):
    """Return (pairs, pre_rows) sampled from one train's feature file."""
    df = df.sort_values(["date", "train_no", "seg_idx"]).reset_index(drop=True)
    n_all = len(df)
    if n_all == 0:
        return None, None, None
    jkey = pd.factorize(df["date"].astype(str) + "|" + df["train_no"].astype(str))[0]
    start = np.flatnonzero(np.r_[True, jkey[1:] != jkey[:-1]])
    lens = np.diff(np.r_[start, n_all])
    jdate = df["date"].to_numpy()[start]

    # ---- pre-departure rows: individual segments
    pre = None
    keep = rng.random(n_all) < p_pre_of_date(df["date"].to_numpy())
    keep &= df["y_arr_delay_B"].notna().to_numpy() & ~df["flag_speed"].astype(bool).to_numpy()
    if keep.any():
        cols = list(dict.fromkeys(pre_feats + ["date", "y_arr_delay_B", "arr_imputed_B",
                                               "seg_idx", "train_no", "to_stn"]))
        pre = df.loc[keep, [c for c in cols if c in df.columns]].copy()

    # ---- live pairs: sample journeys, then P (A, B) pairs per journey
    qj = q_of_date(jdate)
    sel = rng.random(len(start)) < qj
    if not sel.any():
        return None, None, pre
    st, ln = np.repeat(start[sel], P), np.repeat(lens[sel], P)
    i = np.floor(rng.random(len(st)) * ln).astype(np.int64)                # origin segment
    j = i + 1 + np.floor(rng.random(len(st)) * (ln - i)).astype(np.int64)  # stops ahead count
    ai, bj = st + i, st + j - 1                                            # row of A, row ending at B

    pairs = make_pairs(df, ai, bj, st, ln)
    ok = (pairs["y_arr_delay_B"].notna().to_numpy() & pairs["known_delay_A"].notna().to_numpy()
          & ~df["flag_speed"].astype(bool).to_numpy()[bj])
    # validation / test pairs also carry the pre-departure inputs of their target
    # station, so both models can be compared (and combined) on the same pairs
    ev = ok & (pairs["date"].to_numpy(dtype="datetime64[ns]") >= np.datetime64(v0))
    p_tr = pairs[ok & ~ev].reset_index(drop=True)
    p_ev = pairs[ev].reset_index(drop=True)
    if len(p_ev):
        tgt = df.iloc[bj[ev]][pre_feats].reset_index(drop=True)
        tgt.columns = ["tgt__" + c for c in pre_feats]
        for c in tgt.columns:
            if tgt[c].dtype == "float64":
                tgt[c] = tgt[c].astype("float32")
        p_ev = pd.concat([p_ev, tgt], axis=1)
    return p_tr, p_ev, pre


def prepare(df, feats, cats):
    X = pd.DataFrame(index=df.index)
    for c in feats:
        if c in CATEGORICAL:
            X[c] = pd.Categorical(df[c].astype(object).where(df[c].notna(), "<NA>"), categories=cats[c])
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return X


def train_q(name, alpha, Xtr, ytr, Xva, yva, rounds, threads):
    params = dict(PARAMS, objective="quantile", alpha=alpha, metric="quantile", num_threads=threads)
    cat = [c for c in CATEGORICAL if c in Xtr.columns]
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat, params=params, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cat, params=params, free_raw_data=True)
    dtr.construct()
    dva.construct()
    del Xtr, Xva
    t = time.time()
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["val"],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    print(f"  {name}: {m.best_iteration} trees in {time.time() - t:,.0f}s")
    return m


# ------------------------------------------------------------------ metrics
def mae(y, p):
    ok = ~np.isnan(y) & ~np.isnan(p)
    return float(np.mean(np.abs(p[ok] - y[ok]))) if ok.any() else np.nan


def within(y, p, k):
    ok = ~np.isnan(y) & ~np.isnan(p)
    return float(100 * np.mean(np.abs(p[ok] - y[ok]) <= k)) if ok.any() else np.nan


def coverage(y, lo, hi):
    return float(100 * np.mean((y >= lo) & (y <= hi))), float(np.median(hi - lo))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train multi-stop delay models with ranges.")
    ap.add_argument("path", nargs="?", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--val-start", default="2026-07-01")
    ap.add_argument("--test-start", default="2026-08-01")
    ap.add_argument("--test-end", default="2026-09-01")
    ap.add_argument("--pairs-per-journey", type=int, default=3)
    ap.add_argument("--train-pairs", type=int, default=4_000_000)
    ap.add_argument("--train-pre", type=int, default=3_000_000)
    ap.add_argument("--eval-rows", type=int, default=1_000_000, help="cap for validation and test, each")
    ap.add_argument("--range-rows", type=int, default=2_000_000, help="training rows for the p10/p90 models")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=max(2, os.cpu_count() or 2))
    ap.add_argument("--blend-min-rows", type=int, default=BLEND_MIN_ROWS,
                    help="validation pairs a cell needs before its own weight is fitted")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.train_pairs, args.train_pre, args.eval_rows, args.range_rows, args.rounds = \
            800_000, 600_000, 250_000, 400_000, 300
    P = args.pairs_per_journey

    files = [f for f in sorted(glob.glob(os.path.join(args.path, "**", "*.parquet"), recursive=True))
             if not os.path.basename(f).startswith("_")]
    dict_path = os.path.join(args.path, "_data_dictionary.csv")
    if not files or not os.path.exists(dict_path):
        sys.exit(f"Feature files or _data_dictionary.csv not found under {args.path}")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    d = pd.read_csv(dict_path)
    pre_feats = d.loc[~d["column"].isin(EXCLUDE) & d["known_when"].isin(["schedule", "history"]),
                      "column"].tolist()
    have = pd.read_parquet(files[0]).columns
    pre_feats = [c for c in pre_feats if c in have]
    missing = [c for c in READ if c not in have]
    if missing:
        sys.exit(f"Feature files are missing columns {missing}. Re-run build_segments.py and add_network_features.py.")

    # ---- sampling rates from metadata
    import pyarrow.parquet as pq
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    probe = pd.concat([pd.read_parquet(f, columns=["date", "train_no", "seg_idx"])
                       for f in files[:: max(1, len(files) // 60)]])
    dmin, dmax = pd.Timestamp(probe["date"].min()), pd.Timestamp(probe["date"].max())
    seg_per_journey = max(probe.groupby(["date", "train_no"]).size().mean(), 1)
    days = (dmax.date() - dmin.date()).days + 1
    v0, t1, t2 = pd.Timestamp(args.val_start), pd.Timestamp(args.test_start), pd.Timestamp(args.test_end)

    def frac_days(lo, hi):
        lo_d, hi_d = max(lo.date(), dmin.date()), min(hi.date(), dmax.date() + datetime.timedelta(days=1))
        return max((hi_d - lo_d).days, 0) / days
    fr = {"train": frac_days(pd.Timestamp("1900-01-01"), v0), "val": frac_days(v0, t1), "test": frac_days(t1, t2)}
    journeys = total / seg_per_journey
    qj = {k: min(1.0, (args.train_pairs if k == "train" else args.eval_rows) / max(P * journeys * fr[k], 1))
          for k in fr}
    pp = {k: min(1.0, (args.train_pre if k == "train" else args.eval_rows) / max(total * fr[k], 1)) for k in fr}
    v0n, t1n, t2n = np.datetime64(v0), np.datetime64(t1), np.datetime64(t2)

    def rate(dates, r):
        dates = np.asarray(dates, dtype="datetime64[ns]")
        return np.where(dates < v0n, r["train"], np.where(dates < t1n, r["val"],
                        np.where(dates < t2n, r["test"], 0.0)))
    print(f"Files {len(files):,} | ~{journeys:,.0f} journeys | {P} pairs per sampled journey")
    print("Journey sampling: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in qj.items()))

    # ---- build
    def one(k):
        try:
            df = pd.read_parquet(files[k], columns=READ + [c for c in pre_feats if c not in READ])
        except Exception as e:
            return None, None, None, f"{os.path.basename(files[k])}: {type(e).__name__}"
        rng = np.random.default_rng(1000 + k)
        p_tr, p_ev, pre = build_file(df, P, lambda x: rate(x, qj), lambda x: rate(x, pp), rng, v0, t1, pre_feats)
        return p_tr, p_ev, pre, None

    tr_parts, ev_parts, pre_parts, errors = [], [], [], []
    with ThreadPoolExecutor(min(8, args.threads)) as ex:
        for n, (pa, pe, pr, err) in enumerate(ex.map(one, range(len(files))), 1):
            if err:
                errors.append(err)
            if pa is not None and len(pa):
                tr_parts.append(pa)
            if pe is not None and len(pe):
                ev_parts.append(pe)
            if pr is not None and len(pr):
                pre_parts.append(pr)
            if n % 1000 == 0 or n == len(files):
                print(f"  [{n:>6,}/{len(files):,}] pairs {sum(len(x) for x in tr_parts) + sum(len(x) for x in ev_parts):,} | "
                      f"pre rows {sum(len(x) for x in pre_parts):,}  ({time.time() - t0:,.0f}s)")
    if not tr_parts or not ev_parts:
        sys.exit("No training or evaluation pairs. Check --val-start / --test-start against the data dates.")
    pairs = pd.concat(tr_parts + ev_parts, ignore_index=True)
    pre = pd.concat(pre_parts, ignore_index=True)
    del tr_parts, ev_parts, pre_parts

    def split(df):
        dt = df["date"].to_numpy(dtype="datetime64[ns]")
        return (df[dt < v0n].reset_index(drop=True), df[(dt >= v0n) & (dt < t1n)].reset_index(drop=True),
                df[(dt >= t1n) & (dt < t2n)].reset_index(drop=True))
    ptr, pva, pte = split(pairs)
    ptr = ptr.drop(columns=[c for c in ptr.columns if c.startswith("tgt__")])
    rtr, rva, rte = split(pre)
    for name, x in (("train pairs", ptr), ("val pairs", pva), ("test pairs", pte),
                    ("train pre", rtr), ("val pre", rva), ("test pre", rte)):
        if not len(x):
            sys.exit(f"{name} is empty. Check --val-start / --test-start against the data dates.")
    cats = {"train_type": sorted(set(ptr["train_type"].astype(str)) | set(rtr["train_type"].astype(str)) | {"<NA>"}),
            "known_delay_A_src": ["<NA>", "arr", "dep", "none"]}

    # ---- train
    rr = args.range_rows
    live_models, pre_models = {}, {}
    ytr_l = ptr["y_arr_delay_B"] - ptr["known_delay_A"]
    yva_l = pva["y_arr_delay_B"] - pva["known_delay_A"]
    for qn, a in QUANTILES.items():
        n = len(ptr) if qn == "p50" else min(rr, len(ptr))
        print(f"Training LIVE {qn} on {n:,} pairs ...")
        live_models[qn] = train_q(f"live {qn}", a, prepare(ptr.iloc[:n], LIVE_FEATS, cats), ytr_l.iloc[:n],
                                  prepare(pva, LIVE_FEATS, cats), yva_l, args.rounds, args.threads)
        live_models[qn].save_model(os.path.join(args.out, f"live_{qn}.txt"))
    for qn, a in QUANTILES.items():
        n = len(rtr) if qn == "p50" else min(rr, len(rtr))
        print(f"Training PRE-DEPARTURE {qn} on {n:,} rows ...")
        pre_models[qn] = train_q(f"pre {qn}", a, prepare(rtr.iloc[:n], pre_feats, cats),
                                 rtr["y_arr_delay_B"].iloc[:n], prepare(rva, pre_feats, cats),
                                 rva["y_arr_delay_B"], args.rounds, args.threads)
        pre_models[qn].save_model(os.path.join(args.out, f"pre_{qn}.txt"))

    # ---- predict on test
    def live_pred(df):
        X = prepare(df, LIVE_FEATS, cats)
        k = df["known_delay_A"].to_numpy(dtype="float64")
        P_ = np.column_stack([k + live_models[q].predict(X, num_iteration=live_models[q].best_iteration)
                              for q in ("p10", "p50", "p90")])
        P_.sort(axis=1)                                     # keep p10 <= p50 <= p90
        return P_

    def pre_pred(df):
        X = prepare(df, pre_feats, cats)
        P_ = np.column_stack([pre_models[q].predict(X, num_iteration=pre_models[q].best_iteration)
                              for q in ("p10", "p50", "p90")])
        P_.sort(axis=1)
        return P_
    # ---- calibrate the ranges on VALIDATION so that ~80% of actual delays fall inside;
    #      the scale stretches (or shrinks) the distance from p50 to p10 and to p90
    def scale_for(y, P_, target=0.80):
        lo_d, hi_d = P_[:, 1] - P_[:, 0], P_[:, 2] - P_[:, 1]
        for sc in np.arange(0.5, 4.001, 0.02):
            if np.mean((y >= P_[:, 1] - sc * lo_d) & (y <= P_[:, 1] + sc * hi_d)) >= target:
                return float(round(sc, 2))
        return 4.0
    scale_live = scale_for(pva["y_arr_delay_B"].to_numpy(dtype="float64"), live_pred(pva))
    scale_pre = scale_for(rva["y_arr_delay_B"].to_numpy(dtype="float64"), pre_pred(rva))
    print(f"Range scale from validation: live {scale_live}, pre-departure {scale_pre}")

    # ---- combine live and pre-departure: weights per (stops ahead x current delay) cell,
    #      fitted on VALIDATION pairs, then a range scale for the combined prediction
    tcols = ["tgt__" + c for c in pre_feats]

    def pre_on_pairs(df):
        X = df[tcols].copy()
        X.columns = pre_feats
        return pre_pred(X)
    Lv, Pv = live_pred(pva), pre_on_pairs(pva)
    yv = pva["y_arr_delay_B"].to_numpy(dtype="float64")
    si, ki = blend_cells(pva["stops_ahead"], pva["known_delay_A"])
    weights = np.ones((len(BLEND_STOPS_LABELS), len(BLEND_DELAY_LABELS)))
    counts = np.zeros_like(weights, dtype=int)
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    for a_ in range(weights.shape[0]):
        for b_ in range(weights.shape[1]):
            m_ = (si == a_) & (ki == b_)
            counts[a_, b_] = int(m_.sum())
            if m_.sum() >= args.blend_min_rows:
                errs = [np.mean(np.abs(w * Lv[m_, 1] + (1 - w) * Pv[m_, 1] - yv[m_])) for w in grid]
                weights[a_, b_] = float(grid[int(np.argmin(errs))])
    Bv = weights[si, ki][:, None] * Lv + (1 - weights[si, ki][:, None]) * Pv
    Bv.sort(axis=1)
    scale_blend = scale_for(yv, Bv)
    print(f"Blend weights fitted on validation; combined range scale {scale_blend}")

    lp, rp = live_pred(pte), pre_pred(rte)
    pte["p10"], pte["p50"], pte["p90"] = lp[:, 0], lp[:, 1], lp[:, 2]
    rte["p10"], rte["p50"], rte["p90"] = rp[:, 0], rp[:, 1], rp[:, 2]
    for df_, sc in ((pte, scale_live), (rte, scale_pre)):
        df_["lo"] = df_["p50"] - sc * (df_["p50"] - df_["p10"])
        df_["hi"] = df_["p50"] + sc * (df_["p90"] - df_["p50"])
    Pt = pre_on_pairs(pte)
    pte["pre50"], pte["pre_lo"], pte["pre_hi"] = (Pt[:, 1], Pt[:, 1] - scale_pre * (Pt[:, 1] - Pt[:, 0]),
                                                  Pt[:, 1] + scale_pre * (Pt[:, 2] - Pt[:, 1]))
    b50, blo, bhi = apply_blend(lp, Pt, pte["stops_ahead"], pte["known_delay_A"], weights, scale_blend)
    pte["best"], pte["best_lo"], pte["best_hi"] = b50, blo, bhi
    k = pte["known_delay_A"].to_numpy(dtype="float64")
    pte["rule_persist"] = k
    pte["rule_path"] = k + pte["path_hist_median_sum"].to_numpy(dtype="float64")
    rte["rule_hist"] = (rte["hist_arr_delay_mean_30d"].astype("float64")
                        .fillna(rte["stn_B_arr_delay_mean_30d"].astype("float64")).fillna(0))

    # ---- report
    L = []
    out = lambda *a: L.append(" ".join(str(x) for x in a))
    out("=" * 100)
    out("MULTI-STOP MODELS WITH RANGES")
    out("=" * 100)
    out(f"Data : {args.path}")
    out(f"Live pairs   : train {len(ptr):,} | validation {len(pva):,} | test {len(pte):,}")
    out(f"Pre-departure: train {len(rtr):,} | validation {len(rva):,} | test {len(rte):,}")
    out(f"Range models trained on up to {rr:,} rows. Test = {t1.date()} to {t2.date()}, never used in training.")
    out("Trees: " + ", ".join(f"live {q} {m.best_iteration}" for q, m in live_models.items()) + " | " +
        ", ".join(f"pre {q} {m.best_iteration}" for q, m in pre_models.items()))
    out(f"Time : {time.time() - t0:,.0f}s")
    out("")
    out("MAE = mean absolute error of the central (p50) prediction, minutes.")
    out("Range = p10 to p90, then widened/narrowed by a scale fitted on the VALIDATION period so ~80% of")
    out(f"actual delays fall inside (scale: live {scale_live}, pre-departure {scale_pre}). 'In range' is measured on the TEST period and")
    out("should be close to 80%. 'Width' = median width of the range in minutes.")
    out(f"Before scaling, test coverage was {coverage(pte['y_arr_delay_B'].to_numpy(dtype='float64'), pte['p10'].to_numpy(), pte['p90'].to_numpy())[0]:.1f}% (live) and "
        f"{coverage(rte['y_arr_delay_B'].to_numpy(dtype='float64'), rte['p10'].to_numpy(), rte['p90'].to_numpy())[0]:.1f}% (pre-departure).")
    y = pte["y_arr_delay_B"].to_numpy(dtype="float64")

    def live_table(title, groups):
        out("")
        out("-" * 100)
        out(title)
        out("-" * 100)
        out(f"{'group':<14} {'pairs':>10} {'BEST':>7} {'live':>7} {'pre':>7} {'same':>7} "
            f"{'<=5min':>7} {'<=15min':>8} {'in range':>9} {'width':>7}")
        for g in groups.cat.categories if hasattr(groups, "cat") else sorted(groups.unique()):
            m = (groups == g).to_numpy()
            if not m.any():
                continue
            yy = y[m]
            cov, wid = coverage(yy, pte["best_lo"].to_numpy()[m], pte["best_hi"].to_numpy()[m])
            out(f"{str(g):<14} {int(m.sum()):>10,} {mae(yy, pte['best'].to_numpy()[m]):>7.2f} "
                f"{mae(yy, pte['p50'].to_numpy()[m]):>7.2f} {mae(yy, pte['pre50'].to_numpy()[m]):>7.2f} "
                f"{mae(yy, pte['rule_persist'].to_numpy()[m]):>7.2f} "
                f"{within(yy, pte['best'].to_numpy()[m], 5):>6.1f}% {within(yy, pte['best'].to_numpy()[m], 15):>7.1f}% "
                f"{cov:>8.1f}% {wid:>7.1f}")

    out("")
    out("Columns are MAE in minutes on the same test pairs. BEST = live and pre-departure combined with")
    out("weights fitted on the validation month (table below); live = live model alone; pre = pre-departure")
    out("model alone; same = delay stays as it is now. <=5min, <=15min, in range and width refer to BEST.")
    out("")
    out("Combination weights (share given to the LIVE model; the rest goes to pre-departure):")
    out(f"{'stops ahead':<14}" + "".join(f"{'delay ' + d:>14}" for d in BLEND_DELAY_LABELS))
    for a_, lab in enumerate(BLEND_STOPS_LABELS):
        out(f"{lab:<14}" + "".join(f"{weights[a_, b_]:.2f} ({counts[a_, b_]:,})".rjust(14)
                                   for b_ in range(len(BLEND_DELAY_LABELS))))
    out(f"(validation pairs per cell in brackets; cells with fewer than {args.blend_min_rows} keep weight 1.0)")
    out(f"Combined range scale: {scale_blend}")
    allg = pd.Series(["all"] * len(pte))
    live_table("TEST, LIVE: ALL PAIRS", allg)
    sa = pd.cut(pte["stops_ahead"], [0, 1, 3, 6, 10, 20, 1000],
                labels=["1", "2-3", "4-6", "7-10", "11-20", "21+"])
    live_table("TEST, LIVE: BY STOPS AHEAD", sa)
    ta = pd.cut(pte["path_sch_min"], [-1e9, 30, 60, 120, 240, 480, 1e9],
                labels=["<=30 min", "30-60 min", "1-2 h", "2-4 h", "4-8 h", "> 8 h"])
    live_table("TEST, LIVE: BY SCHEDULED TIME AHEAD", ta)
    kb = pd.cut(pte["known_delay_A"], [-1e9, 15, 60, 180, 1e9], labels=["<=15", "16-60", "61-180", ">180"])
    live_table("TEST, LIVE: BY CURRENT DELAY AT A (min)", kb)

    out("")
    out("-" * 100)
    out("TEST, PRE-DEPARTURE (delay at each station, nothing about today known)")
    out("-" * 100)
    yr = rte["y_arr_delay_B"].to_numpy(dtype="float64")
    cov, wid = coverage(yr, rte["lo"].to_numpy(), rte["hi"].to_numpy())
    out(f"{'':<36} {'rows':>10} {'MAE':>7} {'<=5min':>7} {'<=15min':>8} {'in range':>9} {'width':>7}")
    out(f"{'Rule: 30-day avg delay at station':<36} {len(rte):>10,} {mae(yr, rte['rule_hist'].to_numpy()):>7.2f} "
        f"{within(yr, rte['rule_hist'].to_numpy(), 5):>6.1f}% {within(yr, rte['rule_hist'].to_numpy(), 15):>7.1f}%")
    out(f"{'LightGBM pre-departure p50 + range':<36} {len(rte):>10,} {mae(yr, rte['p50'].to_numpy()):>7.2f} "
        f"{within(yr, rte['p50'].to_numpy(), 5):>6.1f}% {within(yr, rte['p50'].to_numpy(), 15):>7.1f}% "
        f"{cov:>8.1f}% {wid:>7.1f}")

    imp = pd.Series(live_models["p50"].feature_importance("gain"), index=live_models["p50"].feature_name())
    imp = (100 * imp / imp.sum()).sort_values(ascending=False)
    out("")
    out("-" * 100)
    out("FEATURE IMPORTANCE, LIVE p50 (share of total gain, top 20)")
    out("-" * 100)
    for f, v in imp.head(20).items():
        out(f"   {f:<28} {v:6.2f}%")
    if errors:
        out("")
        out(f"Read errors ({len(errors)}): " + "; ".join(errors[:10]))

    text = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "_multistop_report.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.out, "feature_lists.json"), "w") as f:
        json.dump({"live": LIVE_FEATS, "pre_departure": pre_feats, "categorical": CATEGORICAL,
                   "categories": cats,
                   "live_target": "y_arr_delay_B (at station B) - known_delay_A (at station A); add known_delay_A back",
                   "pre_target": "y_arr_delay_B", "quantiles": QUANTILES,
                   "range_scale": {"live": scale_live, "pre_departure": scale_pre},
                   "blend": {"stops_edges": BLEND_STOPS_EDGES, "delay_edges": BLEND_DELAY_EDGES,
                             "weights": weights.tolist(), "scale": scale_blend}}, f, indent=2)
    keep = ["date", "train_no", "from_stn", "to_stn", "stops_ahead", "path_sch_min", "known_delay_A",
            "y_arr_delay_B", "p10", "p50", "p90", "lo", "hi", "pre50", "pre_lo", "pre_hi",
            "best", "best_lo", "best_hi", "rule_persist", "arr_imputed_B"]
    te = pte[keep]
    if len(te) > 500_000:
        te = te.sample(500_000, random_state=0)
    te.to_parquet(os.path.join(args.out, "test_predictions.parquet"), index=False)
    print("\n" + text)
    print(f"Saved models and report to {args.out}")


if __name__ == "__main__":
    main()
