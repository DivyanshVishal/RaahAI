#!/usr/bin/env python3
"""
Arrival-delay gap diagnostic
----------------------------
The cleaning log showed that where the SOURCE arrival delay was empty, about 19%
of rows still have a recomputed arrival delay of 2+ minutes. (For departures the
same check passed: empty source delay almost always meant 0-1 min.)

This script compares three groups of arrival rows in the CLEANED data:
    G  source delay EMPTY, computed delay >= 2 min   <- the unexplained rows
    E  source delay EMPTY, computed delay <= 1 min   <- expected blanks (on time / early)
    P  source delay present                          <- normal rows

If G looks like P (same stop positions, delays consistent with the previous
stop), its timestamps are probably real and the recomputed delay can be trusted.
If G is concentrated in particular positions, trains, stations or months, or
its delay jumps away from the neighbouring stops, it is likely bad data.

Read-only. Processes one file at a time, so memory stays low.

Usage:
    python check_arrival_gap.py
    python check_arrival_gap.py processed/clean_data/year=2026 --workers 4

Output: prints the report and saves arrival_gap_report.txt
Requires: pandas, numpy, pyarrow
"""
import argparse
import glob
import os
import sys
import time
import zlib
from collections import Counter
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd

DEFAULT_IN = "processed/clean_data/year=2026"
DEFAULT_OUT = "arrival_gap_report.txt"

COLS = ["date", "train_no", "train_type", "seq", "stop_idx", "n_stops", "station_code",
        "sch_arr", "sch_dep", "act_arr", "act_dep", "delay_arr_min", "delay_dep_min",
        "src_delay_arr_min", "src_delay_dep_min", "reported_at"]
G, E, P = "G", "E", "P"
GROUP_DESC = {G: "source EMPTY, computed >= 2 min",
              E: "source EMPTY, computed <= 1 min",
              P: "source delay present"}
EXAMPLES_PER_FILE = 2
MIN_ROWS_FOR_SHARE = 50
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def mins(x):
    return x.dt.total_seconds() / 60.0


def cut(x, bins, labels):
    out = pd.cut(x, bins, labels=labels).astype(object)
    return out.where(x.notna(), "n/a")


def analyse(path):
    try:
        d = pd.read_parquet(path, columns=COLS)
    except Exception as e:
        return None, None, f"{os.path.basename(path)}: {type(e).__name__}: {str(e)[:80]}"
    if d.empty:
        return None, None, None

    d = d.sort_values(["date", "train_no", "seq"]).reset_index(drop=True)
    arr = d["delay_arr_min"].astype(float)
    dep = d["delay_dep_min"].astype(float)
    src = d["src_delay_arr_min"].astype(float)

    # neighbours within the same journey
    jg = d.groupby(["date", "train_no"], sort=False)
    prev_dep = jg["delay_dep_min"].shift(1).astype(float)
    prev_stn = jg["station_code"].shift(1)
    next_arr = jg["delay_arr_min"].shift(-1).astype(float)

    grp = pd.Series(np.select([src.notna(), arr >= 2, arr < 2], [P, G, E], default=""),
                    index=d.index)
    keep = (arr.notna() & (grp != "")).to_numpy()

    idx = d["stop_idx"].astype(int)
    last = d["n_stops"].astype(int) - 1
    dims = {
        "1. Stop position in journey": pd.Series(
            np.where(idx == 0, "first stop", np.where(idx == last, "last stop", "middle")),
            index=d.index),
        "2. Scheduled dwell at this stop (min)": cut(
            mins(d["sch_dep"] - d["sch_arr"]), [-np.inf, 0.5, 2.5, 5.5, np.inf],
            ["0", "1-2", "3-5", ">5"]),
        "3. Actual dwell at this stop (min)": cut(
            mins(d["act_dep"] - d["act_arr"]), [-np.inf, 0.5, 2.5, 5.5, np.inf],
            ["0", "1-2", "3-5", ">5"]),
        "4. Arrival delay equals departure delay at same stop": pd.Series(
            np.where(dep.isna(), "n/a", np.where((arr - dep).abs() < 0.5, "yes", "no")),
            index=d.index),
        "5. Source DEPARTURE delay at same stop": pd.Series(
            np.where(d["src_delay_dep_min"].isna(), "empty", "present"), index=d.index),
        "6. |arrival delay - previous stop departure delay| (min)": cut(
            (arr - prev_dep).abs(), [-np.inf, 1.5, 5.5, 15.5, 30.5, 60.5, np.inf],
            ["0-1", "2-5", "6-15", "16-30", "31-60", ">60"]),
        "7. |arrival delay - next stop arrival delay| (min)": cut(
            (arr - next_arr).abs(), [-np.inf, 1.5, 5.5, 15.5, 30.5, 60.5, np.inf],
            ["0-1", "2-5", "6-15", "16-30", "31-60", ">60"]),
        "8. reported_at minus actual arrival (min)": cut(
            mins(d["reported_at"] - d["act_arr"]), [-np.inf, 0, 2, 10, 60, np.inf],
            ["<0", "0-2", "2-10", "10-60", ">60"]),
        "9. Computed arrival delay (min)": cut(
            arr, [-np.inf, -5.5, -0.5, 1.5, 5.5, 15.5, 30.5, 60.5, np.inf],
            ["<= -6", "-5..-1", "0..1", "2..5", "6..15", "16..30", "31..60", "> 60"]),
        "10. Month": d["date"].dt.month.map(lambda m: f"{int(m):02d}").astype(object),
        "11. Train type": d["train_type"].astype(object).fillna("n/a"),
    }

    c = Counter()
    gk = grp[keep]
    for name, val in dims.items():
        sizes = pd.DataFrame({"g": gk, "v": val[keep]}).groupby(["g", "v"]).size()
        for (g_, v_), n in sizes.items():
            c[(name, g_, str(v_))] += int(n)
    for (t, g_), n in pd.DataFrame({"t": d["train_no"][keep], "g": gk}).groupby(["t", "g"]).size().items():
        c[("_train", str(t), g_)] += int(n)
    for (s, g_), n in pd.DataFrame({"s": d["station_code"][keep], "g": gk}).groupby(["s", "g"]).size().items():
        c[("_stn", str(s), g_)] += int(n)
    for (t, tt), n in d[["train_no", "train_type"]].astype(str).value_counts().items():
        c[("_ttype", t, tt)] += int(n)

    ex = None
    gi = np.flatnonzero(keep & (grp == G).to_numpy())
    if len(gi):
        rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))
        pick = rng.choice(gi, size=min(EXAMPLES_PER_FILE, len(gi)), replace=False)
        ex = pd.DataFrame({
            "date": d["date"].dt.date.iloc[pick].values,
            "train": d["train_no"].iloc[pick].values,
            "type": d["train_type"].iloc[pick].values,
            "stop": [f"{i + 1}/{n}" for i, n in zip(idx.iloc[pick], d["n_stops"].iloc[pick])],
            "prev_stn": prev_stn.iloc[pick].values,
            "prev_dep_delay": prev_dep.iloc[pick].values,
            "station": d["station_code"].iloc[pick].values,
            "sch_arr": d["sch_arr"].dt.strftime("%H:%M").iloc[pick].values,
            "act_arr": d["act_arr"].dt.strftime("%H:%M").iloc[pick].values,
            "sch_dep": d["sch_dep"].dt.strftime("%H:%M").iloc[pick].values,
            "act_dep": d["act_dep"].dt.strftime("%H:%M").iloc[pick].values,
            "arr_delay": arr.iloc[pick].values,
            "dep_delay": dep.iloc[pick].values,
            "next_arr_delay": next_arr.iloc[pick].values,
        })
    return c, ex, None


def run_map(func, items, workers):
    if workers <= 1:
        for it in items:
            yield func(it)
    else:
        with Pool(workers) as pool:
            for r in pool.imap_unordered(func, items, chunksize=16):
                yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(args.path, "**", "*.parquet"), recursive=True))
             if not os.path.basename(f).startswith("_")]
    if not files:
        sys.exit(f"No cleaned .parquet files found under {args.path}")
    print(f"Files: {len(files):,} | workers: {args.workers}")

    t0 = time.time()
    C, examples, errors = Counter(), [], []
    for i, (c, ex, err) in enumerate(run_map(analyse, files, args.workers), 1):
        if c:
            C.update(c)
        if ex is not None:
            examples.append(ex)
        if err:
            errors.append(err)
        if i % 1000 == 0 or i == len(files):
            print(f"  [{i:>6,}/{len(files):,}]  ({time.time() - t0:,.0f}s)")

    L = []
    out = lambda *a: L.append(" ".join(str(x) for x in a))

    first_dim = "1. Stop position in journey"
    totals = {g: sum(n for (dname, g_, _), n in C.items() if dname == first_dim and g_ == g)
              for g in (G, E, P)}
    all_arr = max(sum(totals.values()), 1)
    out("=" * 78)
    out("ARRIVAL-DELAY GAP DIAGNOSTIC")
    out("=" * 78)
    out(f"Source: {args.path} | files: {len(files):,} | time: {time.time() - t0:,.0f}s")
    out("")
    out("Groups (arrival rows with a computed arrival delay):")
    for g in (G, E, P):
        out(f"  {g}  {GROUP_DESC[g]:<34} {totals[g]:>12,} rows  ({100 * totals[g] / all_arr:5.1f}%)")
    out("")
    out("Each table shows the % of rows WITHIN each group, so columns add up to 100.")
    out("Compare G with P: similar columns suggest G timestamps are as real as normal rows.")

    dim_names = sorted({k[0] for k in C if not k[0].startswith("_")},
                       key=lambda s: int(s.split(".")[0]))
    for name in dim_names:
        rows = {}
        for (dname, g_, v), n in C.items():
            if dname == name:
                rows.setdefault(v, {G: 0, E: 0, P: 0})[g_] += n
        t = pd.DataFrame(rows).T.fillna(0)
        for g in (G, E, P):
            if g not in t.columns:
                t[g] = 0
        pct = (100 * t[[G, E, P]] / t[[G, E, P]].sum().replace(0, np.nan)).round(1)
        pct = pct.sort_values(G, ascending=False)
        if name.startswith("11."):
            pct = pct.head(15)
        else:
            try:
                order = sorted(pct.index, key=lambda v: (v == "n/a", float(v)))
                pct = pct.loc[order]
            except ValueError:
                pass
        out("")
        out("-" * 78)
        out(name)
        out("-" * 78)
        out(pct.rename(columns={G: "G %", E: "E %", P: "P %"}).to_string())

    # concentration
    def conc(prefix, label):
        per = {}
        for (k, key, g_), n in C.items():
            if k == prefix:
                per.setdefault(key, {G: 0, E: 0, P: 0})[g_] += n
        t = pd.DataFrame(per).T.fillna(0)
        if t.empty or t[G].sum() == 0:
            out(f"No G rows by {label}.")
            return t
        t["rows"] = t[[G, E, P]].sum(axis=1)
        t["G_share_%"] = (100 * t[G] / t["rows"]).round(1)
        srt = t[G].sort_values(ascending=False)
        cum = srt.cumsum() / srt.sum()
        n = len(t)
        out(f"{label.capitalize()}s with any G rows: {int((t[G] > 0).sum()):,} of {n:,}")
        for q in (0.01, 0.10):
            k = max(1, int(round(n * q)))
            out(f"  top {int(q * 100)}% of {label}s ({k:,}) hold {100 * cum.iloc[k - 1]:.1f}% of all G rows")
        big = t[(t["rows"] >= MIN_ROWS_FOR_SHARE) & (t["G_share_%"] >= 50)]
        out(f"  {label}s where G is at least half of arrival rows (min {MIN_ROWS_FOR_SHARE} rows): {len(big):,}")
        return t

    out("")
    out("-" * 78)
    out("12. Concentration by train")
    out("-" * 78)
    tt = conc("_train", "train")
    if len(tt) and tt[G].sum():
        types = {}
        for (k, t, ty), n in C.items():
            if k == "_ttype":
                types.setdefault(t, Counter())[ty] += n
        top = tt.sort_values(G, ascending=False).head(12).copy()
        top["type"] = [types.get(i, Counter({"?": 1})).most_common(1)[0][0] for i in top.index]
        out(top[["type", "rows", G, "G_share_%"]].astype({"rows": int, G: int})
            .rename(columns={G: "G_rows"}).to_string())

    out("")
    out("-" * 78)
    out("13. Concentration by station")
    out("-" * 78)
    st = conc("_stn", "station")
    if len(st) and st[G].sum():
        top = st.sort_values(G, ascending=False).head(12)
        out(top[["rows", G, "G_share_%"]].astype({"rows": int, G: int})
            .rename(columns={G: "G_rows"}).to_string())

    out("")
    out("-" * 78)
    out(f"14. Random examples of G rows (delays in minutes)")
    out("-" * 78)
    if examples:
        exdf = pd.concat(examples, ignore_index=True).sort_values(["train", "date", "stop"]).reset_index(drop=True)
        exdf = exdf.sample(n=min(args.examples, len(exdf)), random_state=1)
        out(exdf.to_string(index=False))
    else:
        out("(none)")

    if errors:
        out("")
        out(f"Read errors ({len(errors)}):")
        L.extend(errors[:20])

    text = "\n".join(L) + "\n"
    print("\n" + text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved report to {args.out}")


if __name__ == "__main__":
    freeze_support()
    main()
