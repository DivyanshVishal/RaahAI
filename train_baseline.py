#!/usr/bin/env python3
"""
Baseline delay models
---------------------
Trains two LightGBM models on the feature table and compares them with simple
rules on a time-based split (no shuffling across time).

    LIVE model           the train has left station A, so its current delay is known.
                         Predicts how much the delay changes by B (residual), then
                         adds it to the known delay -> arrival delay at B.
    PRE-DEPARTURE model  nothing about today's running is known yet. Uses only
                         timetable, network and history features -> arrival delay at B.

Default split (edit with --val-start / --test-start):
    train       journeys before 2026-07-01  (January - June, sampled)
    validation  July                        (used for early stopping)
    test        August                      (never touched during training)

Features are chosen from _data_dictionary.csv by when they are known, so the
pre-departure model can never see live or target columns.

Usage:
    pip install lightgbm
    python train_baseline.py --quick              # small trial (~1M rows), roughly 5-10 minutes
    python train_baseline.py                      # full baseline (4M training rows), roughly 30-90 minutes
    python train_baseline.py --train-rows 10000000

Outputs (in --out, default models/baseline):
    live_model.txt, predep_model.txt   LightGBM models
    _model_report.txt                  metrics, baselines, breakdowns, feature importance
    test_predictions.parquet           sample of test rows with predictions, for error analysis
    feature_lists.json                 exact features used by each model

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
DEFAULT_OUT = "models/baseline"

# Never used as model inputs even though they are known in advance:
#  ids / raw codes (the history features already describe the train and section),
#  raw timestamps, and calendar fields the test month has never been seen with.
EXCLUDE = {"date", "train_no", "from_stn", "to_stn", "from_seq", "to_seq",
           "sch_dep_A", "sch_arr_B", "month", "day_of_year"}
CATEGORICAL = ["train_type", "known_delay_A_src"]
KEYS = ["date", "train_no", "from_stn", "to_stn", "seg_idx"]
EXTRA = ["y_arr_delay_B", "known_delay_A", "arr_imputed_B", "flag_speed", "has_target",
         "hist_gain_mean_30d", "sec_all_gain_mean_30d", "hist_arr_delay_mean_30d",
         "stn_B_arr_delay_mean_30d", "train_type", "known_delay_A_src", "seg_idx", "dep_hour",
         "mid_lat", "mid_lng"]

PARAMS = {
    "objective": "l1",            # optimises mean absolute error; robust to extreme delays
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
}


# ---------------------------------------------------------- progress markers
_T0 = time.time()
_STEP = [0]


def step(msg):
    """Print a numbered, timestamped progress marker."""
    _STEP[0] += 1
    print(f"\n[{_STEP[0]:>2}] [{time.time() - _T0:7.1f}s] {msg}", flush=True)


def sub(msg):
    """Print an indented sub-step line (no counter increment)."""
    print(f"      [{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ loading
def pick_features(dictionary_path):
    d = pd.read_csv(dictionary_path)
    ok = ~d["column"].isin(EXCLUDE)
    live = d.loc[ok & d["known_when"].isin(["schedule", "live", "history"]), "column"].tolist()
    pre = d.loc[ok & d["known_when"].isin(["schedule", "history"]), "column"].tolist()
    return live, pre


def compact(df):
    """Shrink memory: 32-bit numbers, Arrow-backed strings."""
    out = {}
    for c in df.columns:
        s = df[c]
        if c == "date" or pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(bool)
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype("float32")
        else:
            try:
                out[c] = s.astype("string[pyarrow]")
            except Exception:
                out[c] = s.astype(object)
    return pd.DataFrame(out, index=df.index)


def load_split(files, cols, splits, probs, threads, seed=0):
    """Read needed columns from every file and keep a random sample per split."""
    def one(i):
        try:
            df = pd.read_parquet(files[i], columns=cols)
        except Exception as e:
            return None, f"{os.path.basename(files[i])}: {type(e).__name__}"
        rng = np.random.default_rng(seed + i)
        out = {}
        for name, (lo, hi) in splits.items():
            m = (df["date"] >= lo) & (df["date"] < hi)
            m &= df["has_target"].astype(bool) & ~df["flag_speed"].astype(bool)
            sub_ = df[m.to_numpy()]
            if probs[name] < 1 and len(sub_):
                sub_ = sub_[rng.random(len(sub_)) < probs[name]]
            if name != "test":                       # ids only needed for the test predictions file
                sub_ = sub_.drop(columns=[c for c in ("train_no", "from_stn", "to_stn") if c in sub_.columns])
            out[name] = compact(sub_)
        return out, None

    parts = {k: [] for k in splits}
    errors = []
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for i, (res, err) in enumerate(ex.map(one, range(len(files))), 1):
            if err:
                errors.append(err)
                continue
            for k in splits:
                if len(res[k]):
                    parts[k].append(res[k])
            if i % 1000 == 0 or i == len(files):
                sizes = " ".join(f"{k}={sum(len(p) for p in parts[k]):,}" for k in splits)
                print(f"  [{i:>6,}/{len(files):,}] {sizes}  ({time.time() - t0:,.0f}s)", flush=True)
    data = {}
    for k in splits:
        data[k] = pd.concat(parts[k], ignore_index=True) if parts[k] else pd.DataFrame(columns=cols)
        parts[k] = None
    return data, errors


def prepare(df, feats, cats):
    X = pd.DataFrame(index=df.index)
    for c in feats:
        if c in CATEGORICAL:
            X[c] = pd.Categorical(df[c].astype(object).where(df[c].notna(), "<NA>"), categories=cats[c])
        else:
            X[c] = df[c].astype("float32")
    return X


# ------------------------------------------------------------------ metrics
def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok], p[ok]
    if not len(y):
        return {"n": 0}
    e = p - y
    return {"n": len(y), "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
            "MedAE": np.median(np.abs(e)), "within5": 100 * np.mean(np.abs(e) <= 5),
            "within15": 100 * np.mean(np.abs(e) <= 15), "bias": np.mean(e)}


def fmt_row(name, m):
    if not m.get("n"):
        return f"{name:<44} (no rows)"
    return (f"{name:<44} {m['n']:>10,} {m['MAE']:>7.2f} {m['RMSE']:>8.2f} {m['MedAE']:>6.2f} "
            f"{m['within5']:>7.1f}% {m['within15']:>7.1f}% {m['bias']:>+7.2f}")


HEADER = (f"{'':<44} {'rows':>10} {'MAE':>7} {'RMSE':>8} {'MedAE':>6} "
          f"{'<=5min':>8} {'<=15min':>8} {'bias':>7}")


def breakdown(out, title, df, key, ycol, pcol, order=None, top=None):
    out("")
    out(title)
    out(f"{'group':<24} {'rows':>10} {'MAE model':>10} {'MAE rule':>9} {'mean |y|':>9}")
    groups = df.groupby(key, observed=True)
    rows = []
    for g, sub_ in groups:
        rows.append((g, len(sub_), np.mean(np.abs(sub_[pcol] - sub_[ycol])),
                     np.mean(np.abs(sub_["rule"] - sub_[ycol])), np.mean(np.abs(sub_[ycol]))))
    if order is not None:
        rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    else:
        rows.sort(key=lambda r: -r[1])
    for g, n, a, b, c in rows[:top] if top else rows:
        out(f"{str(g):<24} {n:>10,} {a:>10.2f} {b:>9.2f} {c:>9.2f}")


def train_one(name, Xtr, ytr, Xva, yva, rounds, threads):
    step(f"Training {name} model "
         f"(train={len(ytr):,} rows, val={len(yva):,} rows, max_rounds={rounds})")
    params = dict(PARAMS, num_threads=threads)
    cat = [c for c in CATEGORICAL if c in Xtr.columns]
    sub("building LightGBM datasets ...")
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat, params=params, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cat, params=params, free_raw_data=True)
    dtr.construct()
    dva.construct()
    del Xtr, Xva
    sub("datasets constructed, starting boosting ...")
    t = time.time()
    model = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
    sub(f"done: {model.best_iteration} trees in {time.time() - t:,.0f}s")
    return model


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train baseline train-delay models.")
    ap.add_argument("path", nargs="?", default=DEFAULT_IN, help="features folder")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--val-start", default="2026-07-01")
    ap.add_argument("--test-start", default="2026-08-01")
    ap.add_argument("--test-end", default="2026-09-01")
    ap.add_argument("--train-rows", type=int, default=4_000_000)
    ap.add_argument("--eval-rows", type=int, default=2_000_000, help="cap for validation and for test")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=max(2, os.cpu_count() or 2))
    ap.add_argument("--quick", action="store_true", help="1M train rows, 300k eval rows, 400 rounds")
    args = ap.parse_args()
    if args.quick:
        args.train_rows, args.eval_rows, args.rounds = 1_000_000, 300_000, 400

    step("Parsed arguments")
    sub(f"path={args.path}")
    sub(f"out={args.out}")
    sub(f"val_start={args.val_start}  test_start={args.test_start}  test_end={args.test_end}")
    sub(f"train_rows={args.train_rows:,}  eval_rows={args.eval_rows:,}  rounds={args.rounds}  threads={args.threads}")

    step("Discovering feature files and data dictionary")
    files = [f for f in sorted(glob.glob(os.path.join(args.path, "**", "*.parquet"), recursive=True))
             if not os.path.basename(f).startswith("_")]
    dict_path = os.path.join(args.path, "_data_dictionary.csv")
    if not files or not os.path.exists(dict_path):
        sys.exit(f"Feature files or _data_dictionary.csv not found under {args.path}")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    sub(f"found {len(files):,} parquet files")
    sub(f"dictionary: {dict_path}")

    step("Selecting feature lists from the data dictionary")
    live_feats, pre_feats = pick_features(dict_path)
    sample_cols = pd.read_parquet(files[0]).columns
    live_feats = [c for c in live_feats if c in sample_cols]
    pre_feats = [c for c in pre_feats if c in sample_cols]
    cols = list(dict.fromkeys(KEYS + EXTRA + live_feats + pre_feats))
    cols = [c for c in cols if c in sample_cols]
    sub(f"live features: {len(live_feats)}")
    sub(f"pre-departure features: {len(pre_feats)}")
    sub(f"total columns to read: {len(cols)}")

    step("Estimating rows on disk and setting sampling rates")
    v0, t1, t2 = (pd.Timestamp(args.val_start), pd.Timestamp(args.test_start), pd.Timestamp(args.test_end))
    splits = {"train": (pd.Timestamp("1900-01-01"), v0), "val": (v0, t1), "test": (t1, t2)}

    import pyarrow.parquet as pq
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    d = pd.read_parquet(files[len(files) // 2], columns=["date"])["date"]
    dmin, dmax = pd.Timestamp(d.min()), pd.Timestamp(d.max())
    for f in files[:: max(1, len(files) // 50)]:
        dd = pd.read_parquet(f, columns=["date"])["date"]
        dmin, dmax = min(dmin, pd.Timestamp(dd.min())), max(dmax, pd.Timestamp(dd.max()))
    days = max((dmax - dmin).days + 1, 1)

    def est(lo, hi):
        lo_d, hi_d = max(lo.date(), dmin.date()), min(hi.date(), dmax.date() + datetime.timedelta(days=1))
        return max(total * max((hi_d - lo_d).days, 0) / days, 1)
    probs = {"train": min(1.0, args.train_rows / est(*splits["train"])),
             "val": min(1.0, args.eval_rows / est(*splits["val"])),
             "test": min(1.0, args.eval_rows / est(*splits["test"]))}
    sub(f"files: {len(files):,}  rows on disk ~{total:,}  data {dmin.date()} -> {dmax.date()}")
    sub("sampling rates: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in probs.items()))

    step("Loading data splits")
    data, errors = load_split(files, cols, splits, probs, min(8, args.threads))
    for k in data:
        if not len(data[k]):
            sys.exit(f"The {k} split is empty. Check --val-start / --test-start against the data dates.")
    tr, va, te = data["train"], data["val"], data["test"]
    sub(f"train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    if errors:
        sub(f"read errors: {len(errors)} (first: {errors[0]})")

    step("Building categorical vocabularies")
    cats = {c: sorted(set(tr[c].astype(object).where(tr[c].notna(), "<NA>").unique()) | {"<NA>"})
            for c in CATEGORICAL if c in tr.columns}
    for c, v in cats.items():
        sub(f"{c}: {len(v)} categories")

    # ---------------- LIVE model: rows where the current delay at A is known
    step("Filtering rows for the LIVE model (known delay at A)")
    def live_rows(df):
        return df[df["known_delay_A"].notna().to_numpy()]
    trL, vaL, teL = live_rows(tr), live_rows(va), live_rows(te)
    sub(f"live rows -> train={len(trL):,}  val={len(vaL):,}  test={len(teL):,}")

    live = train_one("live", prepare(trL, live_feats, cats),
                     (trL["y_arr_delay_B"] - trL["known_delay_A"]).astype("float32"),
                     prepare(vaL, live_feats, cats),
                     (vaL["y_arr_delay_B"] - vaL["known_delay_A"]).astype("float32"),
                     args.rounds, args.threads)
    step("Saving LIVE model")
    live_path = os.path.join(args.out, "live_model.txt")
    live.save_model(live_path)
    sub(f"saved -> {live_path}")

    step("Training PRE-DEPARTURE model")
    pre = train_one("pre-departure", prepare(tr, pre_feats, cats), tr["y_arr_delay_B"].astype("float32"),
                    prepare(va, pre_feats, cats), va["y_arr_delay_B"].astype("float32"),
                    args.rounds, args.threads)
    step("Saving PRE-DEPARTURE model")
    pre_path = os.path.join(args.out, "predep_model.txt")
    pre.save_model(pre_path)
    sub(f"saved -> {pre_path}")

    # ---------------- predictions and rules
    step("Computing predictions and baseline rules on val + test")
    def add_preds(df):
        df = df.copy()
        known = df["known_delay_A"].astype("float64")
        df["pred_live"] = np.where(known.notna(), known + live.predict(prepare(df, live_feats, cats),
                                                                       num_iteration=live.best_iteration), np.nan)
        df["pred_pre"] = pre.predict(prepare(df, pre_feats, cats), num_iteration=pre.best_iteration)
        hist = df["hist_gain_mean_30d"].astype("float64").fillna(df["sec_all_gain_mean_30d"].astype("float64")).fillna(0)
        df["rule_persist"] = known
        df["rule_persist_hist"] = known + hist
        df["rule_pre_hist"] = (df["hist_arr_delay_mean_30d"].astype("float64")
                               .fillna(df["stn_B_arr_delay_mean_30d"].astype("float64")).fillna(0))
        return df
    sub("scoring validation set ...")
    va = add_preds(va)
    sub("scoring test set ...")
    te = add_preds(te)
    sub("predictions ready")

    # ---------------- report
    step("Building report (metrics, baselines, breakdowns, feature importance)")
    L = []
    out = lambda *a: L.append(" ".join(str(x) for x in a))
    out("=" * 100)
    out("BASELINE MODEL REPORT")
    out("=" * 100)
    out(f"Data      : {args.path}")
    out(f"Train     : before {v0.date()} | {len(tr):,} rows sampled ({100 * probs['train']:.1f}%)")
    out(f"Validation: {v0.date()} to {t1.date()} | {len(va):,} rows")
    out(f"Test      : {t1.date()} to {t2.date()} | {len(te):,} rows (never used in training)")
    out(f"Target    : arrival delay at B (minutes). Rows with flag_speed or no target are excluded.")
    out(f"Trees     : live {live.best_iteration}, pre-departure {pre.best_iteration} (early stopping on validation)")
    out(f"Time      : {time.time() - t0:,.0f}s")
    out("")
    out("MAE = mean absolute error (min). MedAE = median absolute error. <=5min / <=15min = share of")
    out("predictions within 5 / 15 minutes of the actual delay. bias = average (prediction - actual).")

    for split_name, df in (("TEST", te), ("VALIDATION", va)):
        dl = df[df["known_delay_A"].notna().to_numpy()]
        dl_clean = dl[~dl["arr_imputed_B"].astype(bool).to_numpy()]
        out("")
        out("-" * 100)
        out(f"{split_name}: LIVE (current delay at A known)")
        out("-" * 100)
        out(HEADER)
        out(fmt_row("Rule: delay at B = delay at A", metrics(dl["y_arr_delay_B"], dl["rule_persist"])))
        out(fmt_row("Rule: delay at A + usual gain on section", metrics(dl["y_arr_delay_B"], dl["rule_persist_hist"])))
        out(fmt_row("LightGBM live model", metrics(dl["y_arr_delay_B"], dl["pred_live"])))
        out(fmt_row("LightGBM live, observed arrivals only", metrics(dl_clean["y_arr_delay_B"], dl_clean["pred_live"])))
        out("")
        out(f"{split_name}: PRE-DEPARTURE (nothing about today's run known)")
        out(HEADER)
        out(fmt_row("Rule: this train's 30-day avg delay at B", metrics(df["y_arr_delay_B"], df["rule_pre_hist"])))
        out(fmt_row("LightGBM pre-departure model", metrics(df["y_arr_delay_B"], df["pred_pre"])))

    # breakdowns on test, live
    step("Computing TEST breakdowns")
    dl = te[te["known_delay_A"].notna().to_numpy()].copy()
    dl["rule"] = dl["rule_persist_hist"]
    out("")
    out("-" * 100)
    out("TEST BREAKDOWNS (live model vs rule 'delay at A + usual gain')")
    out("-" * 100)
    kb = pd.cut(dl["known_delay_A"], [-np.inf, 5, 15, 60, 180, np.inf],
                labels=["<=5", "6-15", "16-60", "61-180", ">180"])
    dl["known_delay_bucket"] = kb.astype(str)
    breakdown(out, "By current delay at A (min):", dl, "known_delay_bucket", "y_arr_delay_B", "pred_live",
              order=["<=5", "6-15", "16-60", "61-180", ">180"])
    breakdown(out, "By train type (largest 12):", dl, "train_type", "y_arr_delay_B", "pred_live", top=12)
    dl["delay_source"] = dl["known_delay_A_src"].astype(str)
    breakdown(out, "By source of the delay at A:", dl, "delay_source", "y_arr_delay_B", "pred_live")
    dl["hour_band"] = pd.cut(dl["dep_hour"], [0, 6, 12, 18, 24], right=False,
                             labels=["00-06", "06-12", "12-18", "18-24"]).astype(str)
    breakdown(out, "By scheduled departure time:", dl, "hour_band", "y_arr_delay_B", "pred_live",
              order=["00-06", "06-12", "12-18", "18-24"])
    dl["region"] = pd.cut(dl["mid_lat"], [0, 15, 20, 25, 40], labels=["south <15N", "15-20N", "20-25N", "north >25N"]).astype(str)
    breakdown(out, "By latitude band of the section:", dl, "region", "y_arr_delay_B", "pred_live",
              order=["south <15N", "15-20N", "20-25N", "north >25N"])

    out("")
    out("By test day (live model). Days with much higher error than usual often point to")
    out("disruptions such as storms or heavy rain; worth checking against weather records:")
    tmp = pd.DataFrame({"day": dl["date"].dt.date.to_numpy(),
                        "ae": (dl["pred_live"] - dl["y_arr_delay_B"]).abs().to_numpy()})
    daily = tmp.groupby("day")["ae"].agg(rows="size", MAE="mean")
    med = daily["MAE"].median()
    for dte, r in daily.iterrows():
        mark = "  <-- high" if r["MAE"] > 1.5 * med else ""
        out(f"   {dte}  rows {int(r['rows']):>8,}  MAE {r['MAE']:6.2f}{mark}")

    step("Computing feature importances")
    for name, model in (("LIVE", live), ("PRE-DEPARTURE", pre)):
        imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
        imp = (100 * imp / imp.sum()).sort_values(ascending=False)
        out("")
        out("-" * 100)
        out(f"FEATURE IMPORTANCE, {name} model (share of total gain, top 25)")
        out("-" * 100)
        for f, v in imp.head(25).items():
            out(f"   {f:<32} {v:6.2f}%")
        low = imp[imp < 0.05].index.tolist()
        if low:
            out(f"   Almost unused (<0.05%): {', '.join(low)}")
        sub(f"{name}: {len(imp)} features, top-1 = {imp.index[0]} ({imp.iloc[0]:.2f}%)")

    if errors:
        out("")
        out(f"Read errors ({len(errors)}): " + "; ".join(errors[:10]))

    step("Writing outputs")
    text = "\n".join(L) + "\n"
    report_path = os.path.join(args.out, "_model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    sub(f"report          -> {report_path}")

    feats_path = os.path.join(args.out, "feature_lists.json")
    with open(feats_path, "w") as f:
        json.dump({"live": live_feats, "pre_departure": pre_feats, "categorical": CATEGORICAL,
                   "live_target": "y_arr_delay_B - known_delay_A (add known_delay_A back)",
                   "pre_target": "y_arr_delay_B", "categories": cats}, f, indent=2)
    sub(f"feature lists   -> {feats_path}")

    keep = KEYS + ["y_arr_delay_B", "known_delay_A", "known_delay_A_src", "train_type", "arr_imputed_B",
                   "pred_live", "pred_pre", "rule_persist", "rule_persist_hist", "rule_pre_hist"]
    te_out = te[keep]
    if len(te_out) > 500_000:
        te_out = te_out.sample(500_000, random_state=0)
    preds_path = os.path.join(args.out, "test_predictions.parquet")
    te_out.to_parquet(preds_path, index=False)
    sub(f"test predictions-> {preds_path}  ({len(te_out):,} rows)")

    step("Done")
    print("\n" + text)
    print(f"Saved models and report to {args.out}")
    print(f"Total wall time: {time.time() - _T0:,.1f}s")


if __name__ == "__main__":
    main()