#!/usr/bin/env python3
"""
Data-flow inspection for designing the daily prediction pipeline
----------------------------------------------------------------
Looks at the project folder (read-only) and writes data_flow_report.txt with:

  1. Machine and Python setup (versions, CPUs, memory, free disk)
  2. Project folders: sizes, file types, when files were last changed
  3. Other scripts in the project (names, imports, functions, and the lines
     that read or write data) - to see how processed/gps_data is produced
  4. The GPS data folder: file layout, what one file contains, when files
     were last written, how fresh the newest data is
  5. Data files changed recently anywhere in the project (is new data arriving?)
  6. How long one prediction takes with the current models
  7. Three questions for your team to answer at the end of the report

Nothing is modified. Secrets are protected: config/.env files are never read,
and anything that looks like a key, token or password is replaced with
<redacted>. Still, please glance through the report before sending it.

Usage (from the project folder, e.g. C:\\GANGBANG):
    python inspect_data_flow.py
    python inspect_data_flow.py --gps processed/gps_data --out data_flow_report.txt
"""
import argparse
import datetime as dt
import glob
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

SKIP_DIRS = {"env", "venv", ".venv", "site-packages", "node_modules", ".git", "__pycache__",
             ".ipynb_checkpoints", ".idea", ".vscode", "Lib", "Scripts", "Include"}
OUR_SCRIPTS = {"profile_train_data.py", "clean_train_data.py", "check_arrival_gap.py",
               "build_segments.py", "add_network_features.py", "train_baseline.py",
               "train_multistop.py", "predict.py", "pick_journeys.py", "inspect_data_flow.py"}
SECRET_FILES = re.compile(r"(^\.env|secret|credential|password|token|\.pem$|\.key$|\.pfx$)", re.I)
DATA_EXT = {".parquet", ".csv", ".json", ".jsonl", ".gz", ".zip", ".xlsx", ".xls", ".txt", ".feather", ".db", ".sqlite"}
CODE_EXT = {".py", ".ipynb", ".bat", ".ps1", ".sh", ".sql"}
INTERESTING = re.compile(r"(read_|to_parquet|to_csv|requests\.|urlopen|http|sqlalchemy|connect\(|cursor|"
                         r"schedule|while True|sleep\(|glob|open\(|makedirs|os\.path\.join|api|fetch|download|"
                         r"datetime\.now|date\.today|cron|task)", re.I)
REDACT_KV = re.compile(r"(?i)((?:api[_-]?key|key|token|secret|password|passwd|pwd|auth|bearer)\s*[=:]\s*)"
                       r"(['\"]?)[^'\"\s,)]+")
REDACT_LONG = re.compile(r"[A-Za-z0-9_\-+=]{24,}")   # key-like runs; '/' excluded so paths survive
REDACT_URLQ = re.compile(r"(https?://[^\s'\"?]+)\?[^\s'\"]*")


def redact(line):
    line = REDACT_KV.sub(lambda m: m.group(1) + m.group(2) + "<redacted>", line)
    line = REDACT_URLQ.sub(lambda m: m.group(1) + "?<redacted>", line)
    # only runs that mix letters and digits look like keys (folder names usually do not)
    return REDACT_LONG.sub(lambda m: "<redacted>" if re.search(r"\d", m.group(0)) and re.search(r"[A-Za-z]", m.group(0))
                           else m.group(0), line)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:,.0f} {u}"
        n /= 1024
    return f"{n:,.1f} PB"


def ts(t):
    return dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


def walk(root, max_depth=6):
    root = os.path.abspath(root)
    base = root.count(os.sep)
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
        if d.count(os.sep) - base >= max_depth:
            dirs[:] = []
        yield d, files


def shape(name):
    s = re.sub(r"\d", "9", name)
    return re.sub(r"[A-Za-z]+", "A", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project folder (default: current folder)")
    ap.add_argument("--gps", default="processed/gps_data", help="raw GPS data folder used by the pipeline")
    ap.add_argument("--models", default="models/multistop")
    ap.add_argument("--features", default="processed/features/year=2026")
    ap.add_argument("--sample-files", type=int, default=300)
    ap.add_argument("--out", default="data_flow_report.txt")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    now = time.time()
    L = []
    out = lambda *a: L.append(" ".join(str(x) for x in a))

    def h(title):
        out("")
        out("=" * 90)
        out(title)
        out("=" * 90)

    out(f"DATA FLOW REPORT  |  generated {dt.datetime.now():%Y-%m-%d %H:%M}  |  project {root}")

    # ------------------------------------------------------------------ 1. machine
    h("1. MACHINE AND PYTHON")
    out(f"OS: {platform.platform()} | Python {sys.version.split()[0]} | CPUs: {os.cpu_count()}")
    for mod in ("pandas", "numpy", "pyarrow", "lightgbm"):
        try:
            m = __import__(mod)
            out(f"  {mod:<10} {getattr(m, '__version__', '?')}")
        except Exception:
            out(f"  {mod:<10} not installed")
    try:
        import psutil
        vm = psutil.virtual_memory()
        out(f"Memory: {human(vm.total)} total, {human(vm.available)} available now")
    except Exception:
        out("Memory: (install psutil to report this; not required)")
    du = shutil.disk_usage(root)
    out(f"Disk of the project drive: {human(du.total)} total, {human(du.free)} free")

    # ------------------------------------------------------------------ 2. folders
    h("2. PROJECT FOLDERS (environment folders skipped)")
    per = defaultdict(lambda: {"n": 0, "size": 0, "ext": Counter(), "new": 0, "old": now})
    for d, files in walk(root, max_depth=4):
        rel = os.path.relpath(d, root)
        key = os.sep.join(rel.split(os.sep)[:3])
        for f in files:
            p = os.path.join(d, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            e = per[key]
            e["n"] += 1
            e["size"] += st.st_size
            e["ext"][os.path.splitext(f)[1].lower() or "(none)"] += 1
            e["new"] = max(e["new"], st.st_mtime)
            e["old"] = min(e["old"], st.st_mtime)
    out(f"{'folder':<46} {'files':>8} {'size':>10}  {'oldest change':<16}  {'newest change':<16}  main types")
    for k in sorted(per, key=lambda x: (x.count(os.sep), x)):
        e = per[k]
        types = ", ".join(f"{x}:{n}" for x, n in e["ext"].most_common(3))
        out(f"{k[:46]:<46} {e['n']:>8,} {human(e['size']):>10}  {ts(e['old']):<16}  {ts(e['new']):<16}  {types}")

    # ------------------------------------------------------------------ 3. scripts
    h("3. OTHER SCRIPTS IN THE PROJECT (how is the data produced?)")
    out("Shows imports, functions and the lines that read / write data or call the network.")
    out("Secrets are redacted; config and .env files are not read.")
    found = 0
    for d, files in walk(root, max_depth=4):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext not in CODE_EXT or f in OUR_SCRIPTS or SECRET_FILES.search(f):
                continue
            p = os.path.join(d, f)
            try:
                if os.path.getsize(p) > 2_000_000:
                    continue
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            found += 1
            if ext == ".ipynb":
                text = "\n".join(re.findall(r'"source":\s*\[(.*?)\]', text, flags=re.S))
                text = text.replace('\\n",', "\n").replace('"', "")
            lines = text.splitlines()
            out("")
            out(f"--- {os.path.relpath(p, root)}  ({len(lines)} lines, changed {ts(os.path.getmtime(p))})")
            imports = sorted({re.sub(r"\s+", " ", l.strip()) for l in lines
                              if re.match(r"\s*(import|from)\s+\w", l)})[:15]
            if imports:
                out("    imports : " + " | ".join(imports))
            defs = [re.sub(r"\(.*", "", l.strip()) for l in lines if re.match(r"\s*(def|class)\s+\w", l)][:20]
            if defs:
                out("    defines : " + ", ".join(defs))
            hits = [redact(l.strip())[:150] for l in lines if INTERESTING.search(l) and not l.strip().startswith("#")]
            for l in hits[:25]:
                out("      " + l)
            if len(hits) > 25:
                out(f"      ... {len(hits) - 25} more lines")
    if not found:
        out("(no other scripts found - the GPS data may come from outside this folder)")

    # ------------------------------------------------------------------ 4. GPS data
    h(f"4. GPS DATA FOLDER: {args.gps}")
    gps = os.path.join(root, args.gps)
    files = sorted(glob.glob(os.path.join(gps, "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        out("Folder not found or empty.")
    else:
        sizes = [os.path.getsize(f) for f in files]
        mt = [os.path.getmtime(f) for f in files]
        rels = [os.path.relpath(f, gps) for f in files]
        out(f"{len(files):,} files, {human(sum(sizes))} total | file sizes: median {human(sorted(sizes)[len(sizes) // 2])}, "
            f"largest {human(max(sizes))}")
        out("Sub-folder layout (first levels): " + ", ".join(
            f"{k} ({n:,})" for k, n in Counter(os.path.dirname(r).split(os.sep)[0] or "." for r in rels).most_common(8)))
        out("File-name patterns (9 = digit, A = letters): " + ", ".join(
            f"{k} ({n:,})" for k, n in Counter(shape(os.path.basename(r)) for r in rels).most_common(6)))
        out("Example names: " + ", ".join(rels[:: max(1, len(rels) // 5)][:6]))
        days = Counter(dt.datetime.fromtimestamp(t).date() for t in mt)
        out(f"When the files were last written ({len(days)} distinct days; most recent first):")
        for d_, n in sorted(days.items(), reverse=True)[:12]:
            out(f"   {d_}  {n:>8,} files")
        if len(days) > 12:
            out(f"   ... {len(days) - 12} earlier days")
        try:
            import pandas as pd
            random.seed(0)
            sample = random.sample(files, min(args.sample_files, len(files)))
            sample = [f for f in sample if f.lower().endswith(".parquet")]
            stats, cols_seen = [], Counter()
            latest_report, latest_date = None, None
            for f in sample:
                try:
                    import pyarrow.parquet as pq
                    names = pq.ParquetFile(f).schema.names
                    cols_seen.update(names)
                    want = [c for c in ("date", "train_no", "reported_at") if c in names]
                    df = pd.read_parquet(f, columns=want)
                except Exception:
                    continue
                s = {"rows": len(df)}
                if "date" in df:
                    dd = pd.to_datetime(df["date"], errors="coerce")
                    s["dates"] = dd.nunique()
                    s["dmin"], s["dmax"] = dd.min(), dd.max()
                    latest_date = s["dmax"] if latest_date is None or s["dmax"] > latest_date else latest_date
                if "train_no" in df:
                    s["trains"] = df["train_no"].nunique()
                if "reported_at" in df:
                    r = pd.to_datetime(df["reported_at"], errors="coerce", utc=True)
                    if r.notna().any():
                        latest_report = r.max() if latest_report is None or r.max() > latest_report else latest_report
                stats.append(s)
            if stats:
                sd = pd.DataFrame(stats)
                out(f"Inside a sample of {len(sd)} files:")
                out(f"   rows per file: median {sd['rows'].median():,.0f}")
                if "trains" in sd:
                    out(f"   trains per file: median {sd['trains'].median():.0f}, max {sd['trains'].max():.0f}"
                        "  (1 = one file per train)")
                if "dates" in sd:
                    out(f"   journey dates per file: median {sd['dates'].median():.0f}, max {sd['dates'].max():.0f}"
                        "  (1 = one file per day)")
                    out(f"   date range in sample: {sd['dmin'].min().date()} -> {sd['dmax'].max().date()}")
                out(f"   columns: {', '.join(c for c, _ in cols_seen.most_common(30))}")
                if latest_date is not None:
                    gap = (pd.Timestamp.now().normalize() - latest_date).days
                    out(f"   newest journey date in sample: {latest_date.date()}  ({gap} days before today)")
                if latest_report is not None:
                    out(f"   newest 'reported_at' in sample: {latest_report}")
        except ImportError:
            out("pandas not available; skipped reading file contents")

    # ------------------------------------------------------------------ 5. recent data
    h("5. DATA FILES CHANGED IN THE LAST 45 DAYS (anywhere in the project)")
    out("Shows whether new data is arriving somewhere, and where. Outputs of our own pipeline")
    out("(processed/clean_data, segments, features, models) are listed too, so you can tell them apart.")
    recent = defaultdict(lambda: [0, 0, 0.0])
    cutoff = now - 45 * 86400
    for d, fs in walk(root, max_depth=6):
        for f in fs:
            if os.path.splitext(f)[1].lower() not in DATA_EXT or SECRET_FILES.search(f):
                continue
            p = os.path.join(d, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_mtime >= cutoff:
                k = os.path.relpath(d, root).split(os.sep)
                k = os.sep.join(k[:3])
                recent[k][0] += 1
                recent[k][1] += st.st_size
                recent[k][2] = max(recent[k][2], st.st_mtime)
    if recent:
        out(f"{'folder':<50} {'files':>8} {'size':>10}  newest change")
        for k, (n, sz, nw) in sorted(recent.items(), key=lambda x: -x[1][2]):
            out(f"{k[:50]:<50} {n:>8,} {human(sz):>10}  {ts(nw)}")
    else:
        out("No data files changed in the last 45 days.")

    # ------------------------------------------------------------------ 6. speed
    h("6. PREDICTION SPEED WITH THE CURRENT MODELS")
    tp = os.path.join(root, args.models, "test_predictions.parquet")
    pred = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict.py")
    if os.path.exists(tp) and os.path.exists(pred):
        try:
            import pandas as pd
            t = pd.read_parquet(tp).sort_values("stops_ahead", ascending=False).iloc[0]
            cmd = [sys.executable, pred, "--train", str(t["train_no"]), "--date",
                   str(pd.Timestamp(t["date"]).date()), "--at", str(t["from_stn"]),
                   "--features", args.features, "--models", args.models]
            for run in (1, 2):
                t0 = time.time()
                res = subprocess.run(cmd, capture_output=True, text=True)
                out(f"predict.py run {run}: {time.time() - t0:.1f} s (exit code {res.returncode})"
                    + ("  <- includes one-time index build" if run == 1 else ""))
            out("Most of this is starting Python and loading the six models; a long-running service")
            out("that keeps the models in memory would predict a whole journey in well under a second.")
        except Exception as e:
            out(f"Could not time predict.py: {type(e).__name__}: {str(e)[:120]}")
    else:
        out("models/multistop/test_predictions.parquet or predict.py not found; skipped.")

    # ------------------------------------------------------------------ 7. questions
    h("7. PLEASE ANSWER (decisions for your team; write below each question)")
    for q in (
        "A. How does NEW data arrive? Who or what creates processed/gps_data (a script, another team,\n"
        "   a manual export)? Does it add new files, append to existing ones, or rewrite everything?",
        "B. How often should predictions update?  (a) once a day before departures  (b) every few\n"
        "   minutes while trains run  (c) both  - and roughly how quickly must a prediction be ready?",
        "C. Where should predictions go?  (a) a CSV / parquet file  (b) a database table (which one?)\n"
        "   (c) an API that an app calls  (d) a dashboard  - and who will use them?",
    ):
        out("")
        out(q)
        out("   Answer: ")

    text = "\n".join(L) + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"Saved {args.out}. Please read it through (especially section 3) before sending.")


if __name__ == "__main__":
    main()
