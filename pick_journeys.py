#!/usr/bin/env python3
"""
Pick journeys to check with predict.py
--------------------------------------
Searches the TEST month of the feature table for a varied set of journeys
(different train types, a very late one, a night one, an on-time one, one where
the departure was not recorded, and the model's worst case), chooses a sensible
station partway along each route, and either prints the predict.py commands or
runs them all and saves the output in one file.

Usage:
    python pick_journeys.py              # print the commands
    python pick_journeys.py --run        # run them all -> journey_checks.txt

Options:
    --features processed/features/year=2026   --models models/multistop
    --test-start 2026-08-01 --test-end 2026-09-01   --seed 7   --out journey_checks.txt

Needs predict.py and train_multistop.py in the same folder.
"""
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

COLS = ["date", "train_no", "train_type", "seg_idx", "from_stn", "known_delay_A",
        "known_delay_A_src", "dep_hour", "y_arr_delay_B"]
PREMIUM = ["RAJ", "SHT", "DRNT", "JSH", "GBR", "AC SUPERFAST", "Humsafar"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="processed/features/year=2026")
    ap.add_argument("--models", default="models/multistop")
    ap.add_argument("--test-start", default="2026-08-01")
    ap.add_argument("--test-end", default="2026-09-01")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run", action="store_true", help="run predict.py on every pick and save the output")
    ap.add_argument("--out", default="journey_checks.txt")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    predict = os.path.join(here, "predict.py")
    if not os.path.exists(predict):
        sys.exit("predict.py must be in the same folder as this script.")

    files = [f for f in sorted(glob.glob(os.path.join(args.features, "**", "*.parquet"), recursive=True))
             if not os.path.basename(f).startswith("_")]
    if not files:
        sys.exit(f"No feature files under {args.features}")
    t0, t1 = pd.Timestamp(args.test_start), pd.Timestamp(args.test_end)
    print(f"Scanning {len(files):,} files for journeys between {t0.date()} and {t1.date()} ...")

    def one(f):
        try:
            d = pd.read_parquet(f, columns=COLS)
        except Exception:
            return None
        d = d[(d["date"] >= t0) & (d["date"] < t1)]
        return d if len(d) else None
    parts = []
    with ThreadPoolExecutor(8) as ex:
        for i, d in enumerate(ex.map(one, files), 1):
            if d is not None:
                parts.append(d)
            if i % 2000 == 0:
                print(f"  {i:,}/{len(files):,}")
    if not parts:
        sys.exit("No journeys found in the test period.")
    df = pd.concat(parts, ignore_index=True)
    df["train_no"] = df["train_no"].astype(str)
    df["train_type"] = df["train_type"].astype(str)
    df["known_delay_A_src"] = df["known_delay_A_src"].astype(str)
    df = df.sort_values(["date", "train_no", "seg_idx"]).reset_index(drop=True)

    g = df.groupby(["date", "train_no"], sort=False)
    J = g.agg(train_type=("train_type", "first"), segments=("seg_idx", "size"),
              max_delay=("known_delay_A", "max"), first_hour=("dep_hour", "first"),
              target_cover=("y_arr_delay_B", lambda s: s.notna().mean()),
              has_arr_src=("known_delay_A_src", lambda s: (s == "arr").any())).reset_index()
    J = J[(J["target_cover"] >= 0.9) & (J["segments"] >= 4)]
    rng = np.random.default_rng(args.seed)
    used = set()

    def choose(mask):
        c = J[mask & ~J["train_no"].isin(used)]
        if c.empty:
            return None
        r = c.iloc[int(rng.integers(len(c)))]
        used.add(r["train_no"])
        return r

    def at_station(r, want_src=None):
        rows = df[(df["date"] == r["date"]) & (df["train_no"] == r["train_no"])].reset_index(drop=True)
        ok = rows["known_delay_A"].notna()
        if want_src:
            ok &= rows["known_delay_A_src"] == want_src
        idx = np.flatnonzero(ok.to_numpy())
        if not len(idx):
            return None
        target = len(rows) // 3
        return rows.loc[idx[np.argmin(np.abs(idx - target))], "from_stn"]

    is_long = J["train_type"].isin(["MEX", "SUF"])
    picks = [
        ("Long-distance express, normal day", is_long & (J["segments"] >= 25) & J["max_delay"].between(15, 60), None),
        ("Premium train", J["train_type"].isin(PREMIUM) & (J["segments"] >= 6), None),
        ("Suburban EMU", (J["train_type"] == "EMU") & (J["segments"] >= 15), None),
        ("Passenger / MEMU", J["train_type"].isin(["PAS", "MEMU", "DEMU"]) & (J["segments"] >= 10), None),
        ("Very late journey (3 h+ at some point)", is_long & (J["max_delay"] >= 180), None),
        ("Night running (starts 00:00-05:00)", (J["first_hour"] < 5) & (J["segments"] >= 10), None),
        ("On-time journey", (J["max_delay"] <= 10) & (J["segments"] >= 10), None),
        ("Departure not recorded at the chosen station", J["has_arr_src"] & (J["segments"] >= 8), "arr"),
    ]
    chosen = []
    for label, mask, src in picks:
        r = choose(mask)
        if r is None:
            print(f"  (no journey found for: {label} - skipped)")
            continue
        st = at_station(r, src)
        if st is None:
            continue
        chosen.append((label, r, st, f"{r['segments'] + 1} stops, worst delay {r['max_delay']:.0f} min"))

    # the live model's worst short-range miss, from the training script's test sample
    tp = os.path.join(args.models, "test_predictions.parquet")
    if os.path.exists(tp):
        t = pd.read_parquet(tp)
        t = t[t["stops_ahead"] <= 5].copy()
        if len(t):
            col = "best" if "best" in t.columns else "p50"     # combined prediction when available
            t["err"] = (t[col] - t["y_arr_delay_B"]).abs()
            w = t.sort_values("err", ascending=False).iloc[0]
            chosen.append(("Worst miss of the prediction (up to 5 stops ahead)",
                           pd.Series({"date": w["date"], "train_no": str(w["train_no"]),
                                      "train_type": "", "segments": 0}),
                           w["from_stn"], f"predicted {w[col]:+.0f}, actual {w['y_arr_delay_B']:+.0f} "
                                          f"at {w['to_stn']} ({int(w['stops_ahead'])} stops ahead)"))

    lines = []
    for k, (label, r, st, note) in enumerate(chosen, 1):
        cmd = ["python", "predict.py", "--train", r["train_no"], "--date", str(pd.Timestamp(r["date"]).date()),
               "--at", st]
        if args.features != "processed/features/year=2026":
            cmd += ["--features", args.features]
        if args.models != "models/multistop":
            cmd += ["--models", args.models]
        lines.append((k, label, f"{r['train_type']} {note}".strip(), cmd))

    print("")
    for k, label, note, cmd in lines:
        print(f"{k}. {label} ({note})")
        print("   " + " ".join(cmd))
    if not args.run:
        print("\nRun these one by one, or run:  python pick_journeys.py --run")
        return

    out = []
    for k, label, note, cmd in lines:
        print(f"Running {k}/{len(lines)}: {label} ...")
        res = subprocess.run([sys.executable, predict] + cmd[2:], capture_output=True, text=True)
        out.append("#" * 108)
        out.append(f"# {k}. {label} ({note})")
        out.append("# " + " ".join(cmd))
        out.append("#" * 108)
        out.append(res.stdout.rstrip())
        if res.returncode:
            out.append(res.stderr.strip()[-800:])
        out.append("")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nSaved all outputs to {args.out}")


if __name__ == "__main__":
    main()
