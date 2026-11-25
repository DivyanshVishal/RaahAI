#!/usr/bin/env python3
"""
Train delay dataset profiler
----------------------------
Inspects the raw station-level dataset and writes a report describing formats,
missing values, duplicates and inconsistencies, so cleaning rules can be designed.

It does NOT modify your data.

Memory-safe: holds roughly --max-rows rows in memory. If the dataset on disk
is larger, a random sample of WHOLE journeys (date + train_no) is drawn from
each file, so stop sequences stay intact. Designed to stay well under 8 GB on
a 16 GB machine.

Usage:
    python profile_train_data.py
    python profile_train_data.py processed/gps_data/year=2026
    python profile_train_data.py --max-rows 500000
    python profile_train_data.py data.csv --samples 5
    python profile_train_data.py data.parquet --dayfirst yes

Output:
    Prints the report and saves it as data_profile_report_2026.txt (or --out).

Requires: pandas, numpy, pyarrow (for .parquet and efficient string dtype).
"""
import argparse
import glob
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)

# --------------------------------------------------------------------------
# CONFIGURE DEFAULT PATH HERE
# --------------------------------------------------------------------------
DEFAULT_DATA_PATH = "processed/gps_data/year=2026"
DEFAULT_REPORT    = "data_profile_report_2026.txt"

# Max rows held in memory. ~2M rows x 24 cols with pyarrow strings is
# roughly 1.5-2.5 GB peak. Raise only if you have headroom.
MAX_ROWS_IN_MEMORY = 2_000_000
# --------------------------------------------------------------------------

EXPECTED_COLS = [
    "date", "train_no", "train_name", "train_type", "seq", "station_code",
    "halts_here", "sch_arrival", "sch_arrival_date", "sch_departure",
    "sch_departure_date", "act_arrival", "act_arrival_date", "act_departure",
    "act_departure_date", "delay_arrival_min", "delay_departure_min", "platform",
    "distance_km", "lat", "lng", "departed", "reported", "reported_at",
]
DATE_COLS = ["date", "sch_arrival_date", "sch_departure_date",
             "act_arrival_date", "act_departure_date"]
TIME_COLS = ["sch_arrival", "sch_departure", "act_arrival", "act_departure", "reported_at"]
BOOL_COLS = ["halts_here", "departed", "reported"]
NUM_COLS = ["seq", "delay_arrival_min", "delay_departure_min", "distance_km", "lat", "lng"]
DT_PAIRS = {  # combined datetime name -> (date col, time col)
    "sch_arrival": ("sch_arrival_date", "sch_arrival"),
    "sch_departure": ("sch_departure_date", "sch_departure"),
    "act_arrival": ("act_arrival_date", "act_arrival"),
    "act_departure": ("act_departure_date", "act_departure"),
}
KEY = ["date", "train_no"]  # one journey
NULL_TOKENS = {"", "na", "n/a", "nan", "nat", "none", "null", "-", "--", "nil", "?"}
# Columns where text like "NIL" or "NAT" may be a real code, not a missing value.
# Only truly empty values count as missing here; look-alikes are reported separately.
TOKEN_EXEMPT_COLS = {"station_code", "train_no", "train_name"}
LOCAL_TZ = "Asia/Kolkata"  # reported_at carries +05:30; other times are assumed local
FAST_FORMATS = [
    "%d %b %Y %H:%M", "%d %b %Y %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
    "%d %b %Y", "%Y-%m-%d", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
]
PANDAS_2 = int(pd.__version__.split(".")[0]) >= 2


# ----------------------------------------------------------------- helpers
class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        text = " ".join(str(p) for p in parts)
        print(text)
        self.lines.append(text)

    def h(self, title):
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def table(self, obj, max_rows=20):
        if obj is None or len(obj) == 0:
            self("(none)")
            return
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        self(obj.head(max_rows).to_string())
        if len(obj) > max_rows:
            self(f"... ({len(obj) - max_rows} more rows)")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def _to_string_df(df):
    """Convert every column to pyarrow-backed string dtype (memory-efficient)."""
    out = {}
    for c in df.columns:
        try:
            out[c] = df[c].astype("string[pyarrow]")
        except Exception:
            try:
                out[c] = df[c].astype("string")
            except Exception:
                out[c] = df[c].astype(str)
    return pd.DataFrame(out, index=df.index)


def _sample_journeys(part, frac, rng):
    """
    Keep a random fraction of whole journeys (date + train_no), never single
    rows, so stop sequences stay intact for the journey-structure checks.
    """
    keys = [c for c in KEY if c in part.columns]
    if not keys:
        k = max(1, int(round(len(part) * frac)))
        return part.iloc[rng.choice(len(part), size=min(k, len(part)), replace=False)]
    jkey = part[keys[0]].astype(str)
    for c in keys[1:]:
        jkey = jkey + "|" + part[c].astype(str)
    uniq = pd.unique(jkey)
    k = max(1, int(round(len(uniq) * frac)))
    if k >= len(uniq):
        return part
    chosen = set(np.asarray(uniq, dtype=object)[rng.choice(len(uniq), size=k, replace=False)])
    return part[jkey.isin(chosen).to_numpy()]


def load(path, max_rows=MAX_ROWS_IN_MEMORY, seed=42):
    """
    Load a single file or a directory of parquet files, memory-safe.

    Never holds more than `max_rows` rows in memory. If the dataset on disk
    exceeds the cap, a random per-file sample is drawn proportional to file
    size. Returns (df, meta).
    """
    rng = np.random.default_rng(seed)

    # -------------------------------------------------- directory of parquet
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        total = len(files)
        if total == 0:
            sys.exit(f"No .parquet files found under: {path}")

        print(f"Found {total} parquet file(s) under {path}")
        print(f"Memory cap: {max_rows:,} rows")
        print("-" * 78)

        # Cheap metadata pass: per-file row counts (no data read).
        try:
            import pyarrow.parquet as pq
            row_counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
            rows_on_disk = int(sum(row_counts))
        except Exception as e:
            print(f"  ! parquet metadata unavailable ({type(e).__name__}); "
                  f"will cap after reading")
            row_counts = None
            rows_on_disk = None

        if rows_on_disk is not None:
            print(f"Rows on disk (from metadata): {rows_on_disk:,}")

        sampled = rows_on_disk is not None and rows_on_disk > max_rows
        frac = (max_rows / rows_on_disk) if sampled else 1.0
        if sampled:
            print(f"Dataset exceeds cap -> sampling ~{frac:.3%} per file")

        frames = []
        ok = failed = 0
        kept_rows = 0
        t0 = time.time()

        for i, f in enumerate(files, 1):
            rel = os.path.relpath(f, path)
            try:
                part = pd.read_parquet(f)
                n = len(part)

                if sampled:
                    part = _sample_journeys(part, frac, rng)

                part = _to_string_df(part)
                frames.append(part)
                kept_rows += len(part)
                ok += 1

                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"[{i:>4}/{total}] OK    {rel:<45} "
                      f"disk={n:>10,}  kept={len(part):>8,}  "
                      f"loaded={ok:<4} failed={failed:<3}  "
                      f"in_mem={kept_rows:>10,}  ({rate:.1f} files/s)")
            except Exception as e:
                failed += 1
                print(f"[{i:>4}/{total}] FAIL  {rel:<45} "
                      f"error={type(e).__name__}: {str(e)[:60]}")

        print("-" * 78)
        print(f"Done in {time.time() - t0:.2f}s  |  loaded={ok}/{total}  |  failed={failed}")
        if not frames:
            sys.exit(f"All parquet reads failed under: {path}")

        df = pd.concat(frames, ignore_index=True)
        frames.clear()
        del frames
        # Re-assert string dtype after concat
        for c in df.columns:
            if not str(df[c].dtype).startswith("string"):
                df[c] = df[c].astype("string[pyarrow]")

        meta = {
            "files_total": total,
            "files_ok": ok,
            "files_failed": failed,
            "rows_on_disk": rows_on_disk,
            "rows_in_memory": len(df),
            "sampled": bool(sampled and rows_on_disk and len(df) < rows_on_disk),
            "source": path,
        }
        print(f"In-memory shape: {df.shape}  (cap={max_rows:,})\n")
        return df, meta

    # -------------------------------------------------- single file
    p = path.lower()
    if p.endswith((".csv", ".txt")):
        df = pd.read_csv(path, dtype="string[pyarrow]",
                         keep_default_na=False, low_memory=False)
    elif p.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", dtype="string[pyarrow]",
                         keep_default_na=False, low_memory=False)
    elif p.endswith((".xlsx", ".xls")):
        df = _to_string_df(pd.read_excel(path, dtype=str, keep_default_na=False))
    elif p.endswith(".parquet"):
        df = _to_string_df(pd.read_parquet(path))
    else:
        sys.exit(f"Unsupported file type: {path}")

    sampled = False
    if len(df) > max_rows:
        df = _sample_journeys(df, max_rows / len(df), rng).reset_index(drop=True)
        sampled = True

    meta = {
        "files_total": 1, "files_ok": 1, "files_failed": 0,
        "rows_on_disk": None, "rows_in_memory": len(df),
        "sampled": sampled, "source": path,
    }
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols  (sampled={sampled})\n")
    return df, meta


def is_null(s, strict=False):
    """True for real nulls and empty strings; also null-like text unless strict."""
    filled = s.fillna("")
    out = s.isna().to_numpy() | (filled == "").to_numpy()
    if not strict:
        out = out | filled.str.lower().isin(NULL_TOKENS).to_numpy()
    return pd.Series(out, index=s.index)


def to_num(s):
    return pd.to_numeric(s.where(~is_null(s)), errors="coerce")


def shape_series(s):
    """Vectorised format pattern: '12:05' -> '99:99', '26 Jan 2026' -> '99 AAA 9999'."""
    out = (s.fillna("").astype(str)
           .str.replace(r"\d", "9", regex=True)
           .str.replace(r"[A-Za-z]", "A", regex=True))
    return out.where(out != "", "<empty>")


def _naive_local(x):
    """Drop timezone (after converting to local time) so all datetimes compare."""
    if getattr(x.dt, "tz", None) is not None:
        return x.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return x


def to_bool_view(s):
    return s.str.lower().map({
        "true": True, "1": True, "yes": True, "y": True, "t": True,
        "false": False, "0": False, "no": False, "n": False, "f": False,
    })


def _to_dt(s, dayfirst):
    """
    Parse strings to naive datetimes. Tries fast fixed formats first, then a
    slow per-value fallback only for whatever is left (keeps 2M rows quick).
    """
    s = s.astype(object).where(s.notna(), None)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    todo = s.notna()
    for fmt in FAST_FORMATS:
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        ok = parsed.notna()
        if ok.any():
            out.loc[parsed.index[ok]] = parsed[ok].astype("datetime64[ns]")
            todo.loc[parsed.index[ok]] = False
    if todo.any():
        kwargs = dict(errors="coerce", dayfirst=dayfirst)
        if PANDAS_2:
            kwargs["format"] = "mixed"
        try:
            rest = _naive_local(pd.to_datetime(s[todo], utc=True, **kwargs))
        except Exception:
            rest = pd.to_datetime(s[todo], **kwargs)
        out.loc[rest.index] = rest.astype("datetime64[ns]")
    return out


def parse_dt(date_s, time_s, dayfirst):
    """Combine a date column and a time column into one datetime."""
    d = date_s.where(~is_null(date_s))
    t = time_s.where(~is_null(time_s))
    combo = d.str.cat(t, sep=" ")
    out = _to_dt(combo, dayfirst)
    # fallback: the time column may already contain a full datetime
    need = out.isna() & t.notna()
    if need.any():
        out.loc[need] = _to_dt(t[need], dayfirst)
    return out


def parse_single(s, dayfirst):
    return _to_dt(s.where(~is_null(s)), dayfirst)


def parse_reported(s):
    """reported_at is ISO with a +05:30 offset: parse as UTC, convert to local, drop tz."""
    kwargs = dict(errors="coerce", utc=True)
    if PANDAS_2:
        kwargs["format"] = "ISO8601"
    try:
        r = pd.to_datetime(s.where(~is_null(s)), **kwargs)
    except Exception:
        kwargs.pop("format", None)
        r = pd.to_datetime(s.where(~is_null(s)), **kwargs)
    return _naive_local(r).astype("datetime64[ns]")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT_DATA_PATH,
                    help=f"File or directory (default: {DEFAULT_DATA_PATH})")
    ap.add_argument("--samples", type=int, default=5, help="example rows per issue")
    ap.add_argument("--dayfirst", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--out", default=DEFAULT_REPORT)
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS_IN_MEMORY,
                    help=f"Cap on rows held in memory (default {MAX_ROWS_IN_MEMORY:,})")
    args = ap.parse_args()
    N = args.samples
    R = Report()

    raw, meta = load(args.path, max_rows=args.max_rows)

    R.h("1. OVERVIEW")
    R("Source:", args.path)
    if meta.get("rows_on_disk") is not None:
        R("Rows on disk:", f"{meta['rows_on_disk']:,}")
    R("Rows analysed:", f"{meta['rows_in_memory']:,}",
      "(SAMPLE)" if meta.get("sampled") else "")
    R("Files loaded:", f"{meta['files_ok']}/{meta['files_total']}",
      f"(failed={meta['files_failed']})" if meta["files_failed"] else "")
    R("Columns:", raw.shape[1])
    if meta.get("sampled"):
        R(f"!! NOTE: dataset exceeded MAX_ROWS_IN_MEMORY={args.max_rows:,}. "
          f"Counts below reflect a random sample of whole journeys (date + train_no), "
          f"not the full dataset. "
          f"Raise --max-rows if you need exact totals.")

    missing_cols = [c for c in EXPECTED_COLS if c not in raw.columns]
    extra_cols = [c for c in raw.columns if c not in EXPECTED_COLS]
    R("Expected columns missing:", missing_cols or "none")
    R("Unexpected extra columns:", extra_cols or "none")
    cols = [c for c in EXPECTED_COLS if c in raw.columns]

    # real nulls (None/NaN in the files) are counted first, then turned into ""
    # so every later check works on plain text
    true_null = raw.isna().sum()
    raw = raw.fillna("")

    # whitespace, then strip
    ws = {c: int((raw[c] != raw[c].str.strip()).sum()) for c in raw.columns}
    ws = {c: n for c, n in ws.items() if n}
    R("Columns with leading/trailing whitespace:", ws or "none")
    df = raw.apply(lambda s: s.str.strip())
    del raw

    # ------------------------------------------------------------- nulls
    R.h("2. MISSING VALUES (true nulls + empty strings + null-like text)")
    R("true_null = None/NaN stored in the files. For station_code, train_no and train_name,")
    R("text such as 'NIL' or 'NAT' may be a real code, so it is listed but NOT counted as missing.")
    rows = []
    for c in df.columns:
        low = df[c].str.lower()
        hits = low[low.isin(NULL_TOKENS)]
        tok = hits[hits != ""].value_counts()
        n_empty = int((df[c] == "").sum())
        n_tok = 0 if c in TOKEN_EXEMPT_COLS else int(tok.sum())
        n_null = n_empty + n_tok
        rows.append({
            "column": c,
            "true_null": int(true_null.get(c, 0)),
            "missing_total": n_null,
            "missing_%": round(100 * n_null / max(len(df), 1), 2),
            ("null_like_text" if c not in TOKEN_EXEMPT_COLS else "null_like_text"):
                (("NOT COUNTED: " if c in TOKEN_EXEMPT_COLS and len(tok) else "")
                 + ", ".join(f"{k}:{v}" for k, v in tok.items())) or "-",
        })
    R.table(pd.DataFrame(rows).set_index("column"), max_rows=60)

    # ------------------------------------------------------------- cardinality
    R.h("3. CARDINALITY AND TOP VALUES")
    for c in df.columns:
        vc = df[c].value_counts()
        top = ", ".join(f"{k!r}:{v}" for k, v in vc.head(5).items())
        R(f"{c:<22} unique={df[c].nunique():<8} top: {top[:150]}")

    # ------------------------------------------------------------- formats
    R.h("4. FORMAT PATTERNS OF DATE / TIME / PLATFORM COLUMNS (9=digit, A=letter)")
    for c in [x for x in DATE_COLS + TIME_COLS + ["platform", "train_no", "seq", "station_code"]
              if x in df.columns]:
        shp = shape_series(df[c])
        pats = shp.value_counts()
        R(f"\n{c}: {len(pats)} distinct patterns")
        for pat, n in pats.head(8).items():
            example = df[c][(shp == pat).to_numpy()].iloc[0]
            R(f"   {pat:<28} {n:>10,}   e.g. {example!r}")
        del shp

    # ------------------------------------------------------------- booleans
    R.h("5. FLAG COLUMNS")
    for c in [x for x in BOOL_COLS if x in df.columns]:
        R(f"\n{c}: raw values")
        R.table(df[c].value_counts(dropna=False))
        unmapped = df[c][to_bool_view(df[c]).isna() & ~is_null(df[c])]
        if len(unmapped):
            R(f"   values not recognisable as true/false: {unmapped.unique()[:10].tolist()}")

    # ------------------------------------------------------------- numerics
    R.h("6. NUMERIC COLUMNS")
    num = {}
    for c in [x for x in NUM_COLS if x in df.columns]:
        n = to_num(df[c])
        num[c] = n
        bad = df[c][n.isna() & ~is_null(df[c])]
        R(f"\n{c}: non-numeric values = {len(bad):,}"
          + (f"  e.g. {bad.unique()[:8].tolist()}" if len(bad) else ""))
        R("   " + n.describe(percentiles=[.01, .05, .5, .95, .99]).round(2).to_string().replace("\n", " | "))
        R(f"   negatives={int((n < 0).sum()):,}  zeros={int((n == 0).sum()):,}  "
          f"non-integer={int((n.notna() & (n % 1 != 0)).sum()):,}")

    # ------------------------------------------------------------- datetime parsing
    R.h("7. DATE / DATETIME PARSING")
    if args.dayfirst == "auto" and "date" in df.columns:
        a = parse_single(df["date"], True)
        b = parse_single(df["date"], False)
        both = a.notna() & b.notna()
        differ = int((both & (a != b)).sum())
        R(f"'date' parse rate  dayfirst=True: {a.notna().mean():.2%}   "
          f"dayfirst=False: {b.notna().mean():.2%}   rows where the two disagree: {differ:,}")
        dayfirst = a.notna().sum() > b.notna().sum() or (
            a.notna().sum() == b.notna().sum() and differ > 0 and
            (a[both].dt.day > 12).sum() == 0 and (b[both].dt.day > 12).sum() > 0)
        R("Chosen dayfirst =", dayfirst, "(override with --dayfirst yes/no if wrong)")
    else:
        dayfirst = args.dayfirst == "yes"
        R("dayfirst =", dayfirst)

    dt = {}
    if "date" in df.columns:
        dt["date"] = parse_single(df["date"], dayfirst)
        R(f"date: parsed {dt['date'].notna().mean():.2%}  range "
          f"{dt['date'].min()} -> {dt['date'].max()}")
    for name, (dc, tc) in DT_PAIRS.items():
        if dc in df.columns and tc in df.columns:
            dt[name] = parse_dt(df[dc], df[tc], dayfirst)
            has_raw = ~is_null(df[tc])
            fail = has_raw & dt[name].isna()
            R(f"{name:<14} raw time present={int(has_raw.sum()):>10,}  parsed={int(dt[name].notna().sum()):>10,}  "
              f"failed={int(fail.sum()):,}"
              + (f"  e.g. {df.loc[fail, [dc, tc]].head(3).values.tolist()}" if fail.any() else ""))
    if "reported_at" in df.columns:
        dt["reported_at"] = parse_reported(df["reported_at"])
        R(f"reported_at    parsed={dt['reported_at'].notna().mean():.2%}  range "
          f"{dt['reported_at'].min()} -> {dt['reported_at'].max()}  (converted to {LOCAL_TZ})")

    # ------------------------------------------------------------- delay recompute
    R.h("8. DOES THE GIVEN DELAY MATCH (actual - scheduled)?")
    show = [c for c in ["date", "train_no", "seq", "station_code"] if c in df.columns]
    for kind in ["arrival", "departure"]:
        gc = f"delay_{kind}_min"
        if f"sch_{kind}" not in dt or f"act_{kind}" not in dt or gc not in num:
            R(f"{kind}: cannot check (columns missing)")
            continue
        comp = (dt[f"act_{kind}"] - dt[f"sch_{kind}"]).dt.total_seconds() / 60
        given = num[gc]
        both = comp.notna() & given.notna()
        diff = (comp - given)[both]
        R(f"\n{kind}: rows with both computed and given delay = {int(both.sum()):,}")
        R(f"   given delay but no computable actual/scheduled = {int((given.notna() & comp.isna()).sum()):,}")
        R(f"   computable but given delay missing             = {int((comp.notna() & given.isna()).sum()):,}")
        if both.any():
            R(f"   |diff| <= 1 min: {(diff.abs() <= 1).mean():.2%}")
            R(f"   diff close to +/-1440 min (date rollover issue): "
              f"{int(((diff.abs() - 1440).abs() <= 5).sum()):,}")
            R("   diff distribution: " + diff.describe().round(1).to_string().replace("\n", " | "))
            worst = diff.abs().sort_values(ascending=False).head(N).index
            ex = df.loc[worst, show + [f"sch_{kind}_date", f"sch_{kind}", f"act_{kind}_date",
                                       f"act_{kind}", gc]].copy()
            ex["computed"] = comp[worst].round(1)
            R("   largest mismatches:")
            R.table(ex)

    # ------------------------------------------------------------- duplicates
    R.h("9. DUPLICATES")
    R("Fully identical rows:", f"{int(df.duplicated().sum()):,}")
    for k in (["date", "train_no", "seq"], ["date", "train_no", "station_code"]):
        if all(c in df.columns for c in k):
            d = df.duplicated(subset=k, keep=False)
            R(f"Rows sharing {k}: {int(d.sum()):,}")
            if d.any():
                R.table(df.loc[d, k + [c for c in ["station_code", "seq", "sch_arrival", "act_arrival",
                                                   "reported_at"] if c in df.columns and c not in k]]
                        .sort_values(k).head(N * 2))

    # ------------------------------------------------------------- journey structure
    if all(c in df.columns for c in KEY) and "seq" in num:
        R.h("10. JOURNEY STRUCTURE (journey = date + train_no)")
        work = df[KEY + [c for c in ["station_code", "halts_here"] if c in df.columns]].copy()
        work["_seq"] = num["seq"]
        for c in ["distance_km"]:
            if c in num:
                work["_dist"] = num[c]
        for name in DT_PAIRS:
            if name in dt:
                work[name] = dt[name]
        work = work.sort_values(KEY + ["_seq"])
        g = work.groupby(KEY, sort=False)
        jt = g["_seq"].agg(["min", "max", "count", "nunique"])
        R(f"Journeys: {len(jt):,}  | stops per journey: "
          + jt["count"].describe().round(1).to_string().replace("\n", " | "))
        R("Starting seq values:", jt["min"].value_counts().head(5).to_dict())
        R(f"Journeys with repeated seq: {int((jt['count'] > jt['nunique']).sum()):,}")
        gaps = jt[(jt["max"] - jt["min"] + 1) > jt["nunique"]]
        R(f"Journeys with gaps in seq: {len(gaps):,}")
        R.table(gaps.head(N))

        # where are the missing times? (origin / middle / destination)
        pos = np.where(g.cumcount() == 0, "first",
                       np.where(g.cumcount(ascending=False) == 0, "last", "middle"))
        work["_pos"] = pos
        R("\nShare of rows with missing value, by stop position:")
        miss = {}
        for c in ["sch_arrival", "sch_departure", "act_arrival", "act_departure"]:
            if c in work.columns:
                miss[c] = work[c].isna().groupby(work["_pos"]).mean().round(3)
        if "delay_arrival_min" in num:
            miss["delay_arrival_min"] = num["delay_arrival_min"].reindex(work.index).isna().groupby(work["_pos"]).mean().round(3)
        if "delay_departure_min" in num:
            miss["delay_departure_min"] = num["delay_departure_min"].reindex(work.index).isna().groupby(work["_pos"]).mean().round(3)
        R.table(pd.DataFrame(miss))

        if "_dist" in work.columns:
            dd = g["_dist"].diff()
            R(f"\nDistance decreasing along seq: {int((dd < 0).sum()):,} rows | "
              f"distance unchanged between stops: {int((dd == 0).sum()):,} rows")
            first_dist = g["_dist"].first()
            R("Distance at first stop (should be 0?):", first_dist.value_counts().head(5).to_dict())

        for kind in ["sch", "act"]:
            a, d = f"{kind}_arrival", f"{kind}_departure"
            if a in work.columns and d in work.columns:
                neg_dwell = (work[d] < work[a]).sum()
                nxt = g[a].shift(-1)
                backwards = (nxt < work[d]).sum()
                R(f"{kind}: departure before arrival at same station = {int(neg_dwell):,} | "
                  f"next station arrival before this departure = {int(backwards):,}")
                dwell = (work[d] - work[a]).dt.total_seconds() / 60
                R(f"   {kind} dwell minutes: " + dwell.describe(percentiles=[.5, .95, .99]).round(1)
                  .to_string().replace("\n", " | "))
                if kind == "sch" and "halts_here" in work.columns:
                    hb = to_bool_view(work["halts_here"])
                    R("   halts_here vs scheduled dwell > 0:")
                    R.table(pd.crosstab(hb.fillna("unknown"), (dwell > 0).map({True: "dwell>0", False: "dwell=0"})
                                        .where(dwell.notna(), "dwell n/a")))

        if "sch_departure" in work.columns and "date" in dt:
            first_idx = g.head(1).index
            off = (work.loc[first_idx, "sch_departure"].dt.normalize() - dt["date"][first_idx]).dt.days
            R("\nOrigin scheduled departure date minus 'date' (days):", off.value_counts().head(5).to_dict())

        # route stability
        if "station_code" in work.columns:
            sig = g["station_code"].agg("|".join)
            per_train = sig.groupby(level="train_no").nunique()
            R(f"\nTrains whose stop list varies across dates: {int((per_train > 1).sum()):,} "
              f"of {len(per_train):,}")
            R.table(per_train[per_train > 1].sort_values(ascending=False).head(N).rename("distinct_routes"))

    # ------------------------------------------------------------- entity consistency
    R.h("11. ENTITY CONSISTENCY")
    if "train_no" in df.columns:
        for c in ["train_name", "train_type"]:
            if c in df.columns:
                n = df.groupby("train_no")[c].nunique()
                R(f"train_no with >1 {c}: {int((n > 1).sum()):,}")
                if (n > 1).any():
                    bad = n[n > 1].index[:N]
                    R.table(df[df["train_no"].isin(bad)].groupby(["train_no", c]).size().rename("rows"))
    if all(c in df.columns for c in ["station_code"]) and "lat" in num and "lng" in num:
        coords = pd.DataFrame({"station_code": df["station_code"],
                               "lat": num["lat"].round(4), "lng": num["lng"].round(4)})
        R(f"Stations: {df['station_code'].nunique():,}")
        R(f"Lat range {num['lat'].min()} .. {num['lat'].max()} | Lng range {num['lng'].min()} .. {num['lng'].max()}")
        R(f"Rows with lat/lng missing: {int((num['lat'].isna() | num['lng'].isna()).sum()):,} | "
          f"exactly 0: {int(((num['lat'] == 0) | (num['lng'] == 0)).sum()):,}")
        multi = coords.dropna().drop_duplicates().groupby("station_code").size()
        R(f"Stations with more than one coordinate pair: {int((multi > 1).sum()):,}")
        R.table(multi[multi > 1].sort_values(ascending=False).head(N).rename("coord_pairs"))
        no_coord = coords.groupby("station_code")["lat"].apply(lambda s: s.isna().all())
        R(f"Stations never having coordinates: {int(no_coord.sum()):,}  e.g. {no_coord[no_coord].index[:10].tolist()}")
        if "station_code" in df.columns:
            R(f"Rows with missing station_code: {int(is_null(df['station_code'], strict=True).sum()):,}")
            look = df["station_code"][df["station_code"].str.lower().isin(NULL_TOKENS - {""})]
            if len(look):
                R("station_code values that look like nulls (check whether they are real codes):")
                lk = df.loc[look.index, ["station_code", "lat", "lng"] +
                            [c for c in ["train_no"] if c in df.columns]]
                R.table(lk.groupby("station_code").agg(
                    rows=("lat", "size"),
                    distinct_coords=("lat", "nunique"),
                    trains=("train_no", "nunique") if "train_no" in lk.columns else ("lat", "size"),
                    example_lat=("lat", "first"), example_lng=("lng", "first")))

    # ------------------------------------------------------------- delay distribution
    R.h("12. DELAY DISTRIBUTION")
    for c in ["delay_arrival_min", "delay_departure_min"]:
        if c in num:
            s = num[c].dropna()
            q = s.quantile([0, .01, .05, .25, .5, .75, .95, .99, .999, 1]).round(1)
            R(f"\n{c}: " + " | ".join(f"p{int(k * 1000) / 10:g}={v}" for k, v in q.items()))
            R(f"   < -30: {int((s < -30).sum()):,}   > 300: {int((s > 300).sum()):,}   "
              f"> 720: {int((s > 720).sum()):,}   > 1440: {int((s > 1440).sum()):,}")
            if "train_type" in df.columns:
                R("   median by train_type:")
                R.table(num[c].groupby(df["train_type"]).agg(["count", "median", "mean"]).round(1)
                        .sort_values("count", ascending=False).head(15))

    # ------------------------------------------------------------- flags vs actuals
    R.h("13. FLAGS vs ACTUAL TIMES")
    if "reported" in df.columns and "act_arrival" in dt:
        R("reported vs act_arrival present:")
        R.table(pd.crosstab(df["reported"], dt["act_arrival"].notna().rename("act_arrival_present")))
    if "departed" in df.columns and "act_departure" in dt:
        R("departed vs act_departure present:")
        R.table(pd.crosstab(df["departed"], dt["act_departure"].notna().rename("act_departure_present")))
    if "reported_at" in dt and "act_arrival" in dt:
        both = dt["reported_at"].notna() & dt["act_arrival"].notna()
        lag = (dt["reported_at"] - dt["act_arrival"]).dt.total_seconds()[both] / 60
        if len(lag):
            R("reported_at minus act_arrival (minutes): "
              + lag.describe(percentiles=[.05, .5, .95]).round(1).to_string().replace("\n", " | "))
            R(f"   reported before the train actually arrived: {int((lag < 0).sum()):,}")

    # ------------------------------------------------------------- coverage
    R.h("14. COVERAGE OVER TIME")
    if "date" in dt:
        d = dt["date"].dropna()
        if len(d):
            all_days = pd.date_range(d.min(), d.max(), freq="D")
            present = set(d.dt.normalize().unique())
            missing_days = [x.date() for x in all_days if x not in present]
            R(f"Days in range: {len(all_days):,} | days with data: {len(present):,} | "
              f"missing days: {len(missing_days):,}  e.g. {missing_days[:10]}")
            per_day = d.dt.normalize().value_counts()
            R("Rows per day: " + per_day.describe().round(1).to_string().replace("\n", " | "))
            if "train_no" in df.columns:
                jp = df.assign(_d=dt["date"]).drop_duplicates(["_d", "train_no"]).groupby("train_no").size()
                R(f"Trains: {len(jp):,} | journeys per train: "
                  + jp.describe().round(1).to_string().replace("\n", " | "))

    R.h("END OF REPORT")
    R.save(args.out)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()