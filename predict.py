#!/usr/bin/env python3
"""
Predict delays for one train journey
------------------------------------
Uses the models from train_multistop.py to predict the arrival delay at every
remaining station of a journey, with a likely range. In live mode the main
answer (BEST) combines the live and pre-departure models with the weights
learned during training.

    Live mode           --at STATION      the train is at (or has just left) STATION.
                                          Its delay there is taken from the data, or
                                          from --delay if you give one.
    Pre-departure mode  (no --at)         nothing about today's run is used; predicts
                                          every station of the journey.

The journey must exist in the feature table (processed/features). For dates in
the past, the actual delays are shown next to the predictions, so this also
works as a backtest tool.

Usage:
    python predict.py --train 12309 --date 2026-08-15 --at CNB
    python predict.py --train 12309 --date 2026-08-15 --at CNB --delay 25
    python predict.py --train 12309 --date 2026-08-15                 # pre-departure
    python predict.py --train 12309 --date 2026-08-15 --at CNB --csv out.csv

Requires train_multistop.py in the same folder, plus pandas, numpy, pyarrow, lightgbm.
"""
import argparse
import datetime
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import lightgbm as lgb
    from train_multistop import make_pairs, prepare, apply_blend, LIVE_FEATS
except ImportError as e:
    sys.exit(f"Could not import a requirement ({e}). Keep predict.py next to train_multistop.py "
             f"and install lightgbm.")

DEFAULT_FEATURES = "processed/features/year=2026"
DEFAULT_MODELS = "models/multistop"


def train_index(feat_dir):
    """train_no -> feature file, cached in _train_index.csv inside the features folder."""
    path = os.path.join(feat_dir, "_train_index.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str)
    files = [f for f in sorted(glob.glob(os.path.join(feat_dir, "**", "*.parquet"), recursive=True))
             if not os.path.basename(f).startswith("_")]
    print(f"Building train index over {len(files):,} files (first run only) ...")

    def one(f):
        try:
            t = pd.read_parquet(f, columns=["train_no"])["train_no"].astype(str).unique()
            return [(x, os.path.relpath(f, feat_dir)) for x in t]
        except Exception:
            return []
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, files):
            rows.extend(r)
    idx = pd.DataFrame(rows, columns=["train_no", "file"])
    try:
        idx.to_csv(path, index=False)
    except OSError:
        pass
    return idx


def load_models(model_dir):
    meta_path = os.path.join(model_dir, "feature_lists.json")
    if not os.path.exists(meta_path):
        sys.exit(f"{meta_path} not found. Run train_multistop.py first.")
    meta = json.load(open(meta_path))
    models = {}
    for kind in ("live", "pre"):
        for q in ("p10", "p50", "p90"):
            p = os.path.join(model_dir, f"{kind}_{q}.txt")
            if not os.path.exists(p):
                sys.exit(f"Model file missing: {p}")
            models[(kind, q)] = lgb.Booster(model_file=p)
    return meta, models


def raw_q(models, kind, X):
    P = np.column_stack([models[(kind, q)].predict(X) for q in ("p10", "p50", "p90")])
    P.sort(axis=1)
    return P


def predict_q(models, kind, X, scale):
    P = raw_q(models, kind, X)
    lo = P[:, 1] - scale * (P[:, 1] - P[:, 0])
    hi = P[:, 1] + scale * (P[:, 2] - P[:, 1])
    return P[:, 1], lo, hi


def fmin(x):
    """Whole minutes with a sign; never prints -0."""
    return f"{int(round(float(x))):+d}"


def fmt_time(ts):
    return "" if pd.isna(ts) else pd.Timestamp(ts).strftime("%d %b %H:%M")


def main():
    ap = argparse.ArgumentParser(description="Predict delays for one train journey.")
    ap.add_argument("--train", required=True, help="train number")
    ap.add_argument("--date", required=True, help="journey start date, YYYY-MM-DD")
    ap.add_argument("--at", default="", help="station code where the train is now (live mode)")
    ap.add_argument("--delay", type=float, default=None, help="current delay at --at in minutes (optional)")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--csv", default="", help="also save the table to this CSV file")
    args = ap.parse_args()

    meta, models = load_models(args.models)
    cats = meta["categories"]
    scale = meta.get("range_scale", {"live": 1.0, "pre_departure": 1.0})
    live_feats, pre_feats = meta["live"], meta["pre_departure"]
    if live_feats != LIVE_FEATS:
        print("Note: models were trained with a different feature list than this train_multistop.py; "
              "using the list saved with the models.")

    idx = train_index(args.features)
    hit = idx[idx["train_no"] == str(args.train)]
    if hit.empty:
        sys.exit(f"Train {args.train} not found in {args.features}")
    df = pd.read_parquet(os.path.join(args.features, hit["file"].iloc[0]))
    df = df[df["train_no"].astype(str) == str(args.train)]
    day = pd.Timestamp(args.date)
    J = df[df["date"] == day].sort_values(["date", "train_no", "seg_idx"]).reset_index(drop=True)
    if J.empty:
        dates = sorted(pd.to_datetime(df["date"]).dt.date.unique())
        near = sorted(dates, key=lambda d: abs((pd.Timestamp(d) - day).days))[:5]
        sys.exit(f"No journey of train {args.train} starting {day.date()}. Nearest dates: "
                 + ", ".join(str(d) for d in sorted(near)))
    n = len(J)
    stations = [J["from_stn"].iloc[0]] + J["to_stn"].astype(str).tolist()

    base = pd.DataFrame({
        "stop": np.arange(1, n + 1),
        "station": J["to_stn"].astype(str),
        "sch_arrival": J["sch_arr_B"],
        "actual_delay": pd.to_numeric(J["y_arr_delay_B"], errors="coerce"),
    })

    # ---- pre-departure: every station
    Xp = prepare(J, pre_feats, cats)
    P_pre = raw_q(models, "pre", Xp)
    sp = scale.get("pre_departure", 1.0)
    base["pre_delay"] = P_pre[:, 1]
    base["pre_low"] = P_pre[:, 1] - sp * (P_pre[:, 1] - P_pre[:, 0])
    base["pre_high"] = P_pre[:, 1] + sp * (P_pre[:, 2] - P_pre[:, 1])

    title = f"Train {args.train}, journey starting {day.date()} ({J['train_type'].iloc[0]}), {n + 1} stops"
    if args.at:
        code = args.at.strip().upper()
        pos = [k for k, s in enumerate(J["from_stn"].astype(str).str.upper()) if s == code]
        if not pos:
            if code == stations[-1].upper():
                sys.exit(f"{code} is the last stop of this journey; there is nothing ahead to predict.")
            sys.exit(f"{code} is not a stop of this journey. Stops: " + " ".join(stations))
        i = pos[0]
        Jl = J.copy()
        if args.delay is not None:
            Jl.loc[i, ["known_delay_A", "dep_delay_A"]] = float(args.delay)
            Jl.loc[i, "known_delay_A_src"] = "dep"
        known = pd.to_numeric(Jl.loc[i, "known_delay_A"], errors="coerce")
        if pd.isna(known):
            sys.exit(f"No recorded delay at {code} for this journey. Pass it with --delay MINUTES.")
        ahead = np.arange(i, n)
        pairs = make_pairs(Jl, np.full(len(ahead), i), ahead, np.zeros(len(ahead)), np.full(len(ahead), n))
        Xl = prepare(pairs, live_feats, cats)
        k = float(known)
        P_live = k + raw_q(models, "live", Xl)
        sl = scale.get("live", 1.0)
        base = base.iloc[ahead].reset_index(drop=True)
        base.insert(1, "stops_ahead", np.arange(1, len(base) + 1))
        base["live_delay"] = P_live[:, 1]
        base["live_low"] = P_live[:, 1] - sl * (P_live[:, 1] - P_live[:, 0])
        base["live_high"] = P_live[:, 1] + sl * (P_live[:, 2] - P_live[:, 1])
        blend = meta.get("blend")
        if blend:
            b50, blo, bhi = apply_blend(P_live, P_pre[ahead], base["stops_ahead"].to_numpy(),
                                        np.full(len(base), k), blend["weights"], blend["scale"])
        else:                                   # models trained before blending existed
            b50, blo, bhi = base["live_delay"].to_numpy(), base["live_low"].to_numpy(), base["live_high"].to_numpy()
        base["best_delay"], base["best_low"], base["best_high"] = b50, blo, bhi
        src = "given with --delay" if args.delay is not None else f"recorded ({Jl.loc[i, 'known_delay_A_src']})"
        title += f"\nAt {code}: current delay {k:+.0f} min, {src}. Predicting the next {len(base)} stops."

    # ---- print
    print("=" * 108)
    print(title)
    print("=" * 108)
    live = "live_delay" in base.columns
    head = f"{'stop':>4} {'station':<8} {'sch arrival':<13}"
    if live:
        head += f" {'ahead':>5} {'BEST delay':>10} {'range':>13} {'exp. arrival':<13} {'live':>5} {'pre':>5} {'actual':>7}"
    else:
        head += f" {'PRE delay':>9} {'range':>13} {'actual':>7}"
    print(head)
    print("-" * len(head))
    for r in base.itertuples():
        line = f"{r.stop:>4} {r.station:<8} {fmt_time(r.sch_arrival):<13}"
        if live:
            exp = (pd.Timestamp(r.sch_arrival).to_pydatetime() + datetime.timedelta(minutes=float(r.best_delay))
                   if not pd.isna(r.sch_arrival) else pd.NaT)
            line += (f" {r.stops_ahead:>5} {fmin(r.best_delay):>10} {f'{fmin(r.best_low)} to {fmin(r.best_high)}':>13}"
                     f" {fmt_time(exp):<13} {fmin(r.live_delay):>5} {fmin(r.pre_delay):>5}")
        else:
            line += f" {fmin(r.pre_delay):>9} {f'{fmin(r.pre_low)} to {fmin(r.pre_high)}':>13}"
        line += f" {fmin(r.actual_delay):>7}" if not pd.isna(r.actual_delay) else f" {'-':>7}"
        print(line)
    if base["actual_delay"].notna().any():
        m = base["actual_delay"].notna()
        err = lambda col: np.mean(np.abs(base.loc[m, col] - base.loc[m, "actual_delay"]))
        if live:
            inr = ((base.loc[m, "actual_delay"] >= base.loc[m, "best_low"])
                   & (base.loc[m, "actual_delay"] <= base.loc[m, "best_high"])).mean()
            msg = (f"\nBacktest on this journey: BEST mean error {err('best_delay'):.1f} min "
                   f"(live alone {err('live_delay'):.1f}, pre-departure alone {err('pre_delay'):.1f}), "
                   f"{100 * inr:.0f}% of actuals inside the BEST range")
        else:
            inr = ((base.loc[m, "actual_delay"] >= base.loc[m, "pre_low"])
                   & (base.loc[m, "actual_delay"] <= base.loc[m, "pre_high"])).mean()
            msg = (f"\nBacktest on this journey: PRE mean error {err('pre_delay'):.1f} min, "
                   f"{100 * inr:.0f}% of actuals inside the range")
        print(msg)
    print("Delays in minutes (+ = late). Ranges aim to contain about 80% of actual outcomes.")
    if live:
        print("BEST = live and pre-departure predictions combined (weights learned in training); "
              "live / pre = each model alone.")
    if args.csv:
        base.to_csv(args.csv, index=False)
        print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
