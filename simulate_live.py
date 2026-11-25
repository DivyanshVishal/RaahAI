#!/usr/bin/env python3
"""
Live replay simulator
---------------------
There is no real-time feed, so this script REPLAYS real journeys from the data
as if they were happening now, and sends updated predictions to the backend.

How it works
    1. Picks random trains that ran on one day of the TEST month (default August
       2026). The models never saw that month in training, so every prediction is
       a genuine forecast.
    2. Builds each train's real events (departed station X, arrived at station Y)
       and replays them in time order on a simulated clock (default 60x: one real
       second = one simulated minute; a whole day takes 24 minutes).
    3. Every event updates that train's predictions for all remaining stations
       straight away. Before a train starts, a pre-departure forecast is sent.
    4. Updates are collected and POSTed as one JSON batch every --flush seconds
       (default 2) to the Node.js backend.
    5. When a train reaches a station, the latest prediction made for that station
       is compared with what actually happened ("check"), and a running accuracy
       score is kept - this is what shows the concept working.

Future information is never sent: an update only contains actual times for
stations the train has already reached at that moment of the simulation.

Usage (from the project folder):
    python simulate_live.py                                   # 30 random trains, 2026-08-20, 60x
    python simulate_live.py --trains 50 --date 2026-08-14 --speed 120
    python simulate_live.py --start-time 07:00 --hours 3      # replay only 07:00-10:00
    python simulate_live.py --train-list 12309 12951          # specific trains
    python simulate_live.py --dry-run --fast                  # no backend, no waiting: writes sim_log.jsonl
    python simulate_live.py --url http://localhost:3000/api/predictions

Messages sent (JSON, POST, Content-Type: application/json):
    {"type": "simulation_start", ...}   once: settings + every simulated journey with its stops
    {"type": "batch", "updates": [...], "stats": {...}, ...}   every --flush seconds
    {"type": "simulation_end", "stats": {...}}   once at the end
A full example of each is written to sim_log.jsonl.

Needs predict.py and train_multistop.py in the same folder.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from predict import load_models, raw_q
    from train_multistop import make_pairs, prepare, apply_blend
except ImportError as e:
    sys.exit(f"Could not import predict.py / train_multistop.py ({e}). Keep this script next to them.")

TZ = "+05:30"                       # the data's times are Indian Standard Time
ARR_UNKNOWN = ["dep_delay_A", "dwell_overrun_A", "trend_2stops", "max_dep_delay_so_far"]


# ------------------------------------------------------------------ helpers
def iso(ts):
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S") + TZ


def num(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, nd)


def add_min(ts, m):
    """Timestamp + minutes, using plain datetime arithmetic (no warnings on any pandas version)."""
    return pd.Timestamp(pd.Timestamp(ts).to_pydatetime() + dt.timedelta(minutes=float(m)))


def plus_min(ts, m):
    if ts is None or pd.isna(ts) or m is None or pd.isna(m):
        return None
    return add_min(ts, m)


class Sender:
    """POSTs JSON to the backend; never crashes the demo. Failed messages go to an outbox file."""

    def __init__(self, url, dry_run, log_path, outbox_path, timeout=5.0, retries=2):
        self.url, self.dry, self.timeout, self.retries = url, dry_run, timeout, retries
        self.log = open(log_path, "w", encoding="utf-8")
        self.outbox_path = outbox_path
        self.sent = self.failed = 0

    def send(self, msg):
        body = json.dumps(msg, ensure_ascii=False, allow_nan=False)
        self.log.write(body + "\n")
        self.log.flush()
        if self.dry:
            return True
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=body.encode("utf-8"), method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if 200 <= r.status < 300:
                        self.sent += 1
                        return True
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(0.3 * (attempt + 1))
        self.failed += 1
        with open(self.outbox_path, "a", encoding="utf-8") as f:
            f.write(body + "\n")
        if self.failed in (1, 10) or self.failed % 50 == 0:
            print(f"  ! backend not reachable at {self.url} ({self.failed} batches saved to {self.outbox_path})")
        return False

    def close(self):
        self.log.close()


# ------------------------------------------------------------------ journeys
def find_journeys(feat_dir, day, threads):
    """(train_no, file) of every journey starting on `day`. Cached per date."""
    cache = os.path.join(feat_dir, f"_sim_journeys_{day.date()}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype=str)
    files = [f for f in sorted(glob.glob(os.path.join(feat_dir, "**", "*.parquet"), recursive=True))
             if not os.path.basename(f).startswith("_")]
    print(f"Finding journeys on {day.date()} in {len(files):,} files (first run for this date only) ...")

    def one(f):
        try:
            d = pd.read_parquet(f, columns=["date", "train_no", "seg_idx", "y_arr_delay_B"])
        except Exception:
            return []
        d = d[d["date"] == day]
        if d.empty:
            return []
        g = d.groupby("train_no")
        return [(str(t), os.path.relpath(f, feat_dir), int(len(x)), float(x["y_arr_delay_B"].notna().mean()))
                for t, x in g]
    rows = []
    with ThreadPoolExecutor(threads) as ex:
        for r in ex.map(one, files):
            rows.extend(r)
    out = pd.DataFrame(rows, columns=["train_no", "file", "segments", "cover"]).astype(str)
    try:
        out.to_csv(cache, index=False)
    except OSError:
        pass
    return out


class Journey:
    def __init__(self, jid, J, name, ctx):
        self.jid, self.J, self.n = jid, J.reset_index(drop=True), len(J)
        self.ctx = ctx
        self.train_no = str(J["train_no"].iloc[0])
        self.train_type = str(J["train_type"].iloc[0])
        self.name = name
        self.date = pd.Timestamp(J["date"].iloc[0])
        Xp = prepare(self.J, ctx["pre_feats"], ctx["cats"])
        self.P_pre = raw_q(ctx["models"], "pre", Xp)
        self.last_pred = {}             # row -> (best, lo, hi, made_at, mode)
        self.completed = []             # stations already reached (only known facts)
        self.current_delay = None

    def stations(self):
        J = self.J
        out = [{"stop_no": 0, "station": str(J["from_stn"].iloc[0]), "lat": num(J["from_lat"].iloc[0], 5),
                "lng": num(J["from_lng"].iloc[0], 5), "scheduled_arrival": None,
                "scheduled_departure": iso(J["sch_dep_A"].iloc[0])}]
        for r in range(self.n):
            out.append({"stop_no": r + 1, "station": str(J["to_stn"].iloc[r]), "lat": num(J["to_lat"].iloc[r], 5),
                        "lng": num(J["to_lng"].iloc[r], 5), "scheduled_arrival": iso(J["sch_arr_B"].iloc[r]),
                        "scheduled_departure": iso(J["sch_dep_A"].iloc[r + 1]) if r + 1 < self.n else None})
        return out

    def events(self):
        """Real events: departure from A of row i, arrival at B of row i."""
        J, ev = self.J, []
        for i in range(self.n):
            d = num(J["dep_delay_A"].iloc[i], 3)
            if d is not None:
                ev.append((plus_min(J["sch_dep_A"].iloc[i], d), 1, self.jid, i, "departure", d))
            a = num(J["y_arr_delay_B"].iloc[i], 3)
            if a is not None:
                ev.append((plus_min(J["sch_arr_B"].iloc[i], a), 0, self.jid, i, "arrival", a))
        return [e for e in ev if e[0] is not None]

    def first_time(self):
        return pd.Timestamp(self.J["sch_dep_A"].iloc[0])

    # ---------------------------------------------------------- predictions
    def _pre(self, rows):
        sp = self.ctx["scale"].get("pre_departure", 1.0)
        P = self.P_pre[rows]
        return P[:, 1], P[:, 1] - sp * (P[:, 1] - P[:, 0]), P[:, 1] + sp * (P[:, 2] - P[:, 1])

    def predict_live(self, i, mode, arr_delay=None):
        """Predictions for rows i..n-1 (arrivals at stations after A_i), from what is known now."""
        ctx, J = self.ctx, self.J
        Jl = J.copy()
        if mode == "arrival":           # train has arrived at A_i but not left: departure facts unknown
            for c in ARR_UNKNOWN:
                if c in Jl.columns:
                    Jl.loc[i, c] = np.nan
            if arr_delay is not None:   # the delay observed in the arrival event itself
                Jl.loc[i, "arr_delay_A"] = arr_delay
            Jl.loc[i, "known_delay_A"] = Jl.loc[i, "arr_delay_A"]
            Jl.loc[i, "known_delay_A_src"] = "arr"
        k = pd.to_numeric(Jl.loc[i, "known_delay_A"], errors="coerce")
        if pd.isna(k):
            return None
        ahead = np.arange(i, self.n)
        pairs = make_pairs(Jl, np.full(len(ahead), i), ahead, np.zeros(len(ahead)), np.full(len(ahead), self.n))
        P_live = float(k) + raw_q(ctx["models"], "live", prepare(pairs, ctx["live_feats"], ctx["cats"]))
        sl = ctx["scale"].get("live", 1.0)
        live50 = P_live[:, 1]
        blend = ctx["blend"]
        if blend:
            b50, blo, bhi = apply_blend(P_live, self.P_pre[ahead], np.arange(1, len(ahead) + 1),
                                        np.full(len(ahead), float(k)), blend["weights"], blend["scale"])
        else:
            b50 = live50
            blo, bhi = live50 - sl * (live50 - P_live[:, 0]), live50 + sl * (P_live[:, 2] - live50)
        pre50, _, _ = self._pre(ahead)
        return float(k), ahead, b50, blo, bhi, live50, pre50

    def prediction_rows(self, ahead, b50, blo, bhi, live50, pre50, sim_now, mode):
        J, out = self.J, []
        for n_, r in enumerate(ahead):
            sch = J["sch_arr_B"].iloc[r]
            out.append({
                "stop_no": int(r) + 1, "station": str(J["to_stn"].iloc[r]),
                "lat": num(J["to_lat"].iloc[r], 5), "lng": num(J["to_lng"].iloc[r], 5),
                "scheduled_arrival": iso(sch), "stops_ahead": n_ + 1,
                "predicted_delay_min": num(b50[n_]), "range_low_min": num(blo[n_]), "range_high_min": num(bhi[n_]),
                "expected_arrival": iso(plus_min(sch, b50[n_])),
                "live_model_delay_min": num(live50[n_]) if live50 is not None else None,
                "pre_departure_delay_min": num(pre50[n_]),
            })
            self.last_pred[int(r)] = (float(b50[n_]), float(blo[n_]), float(bhi[n_]), sim_now, mode)
        return out

    def train_info(self):
        return {"train_no": self.train_no, "train_name": self.name, "train_type": self.train_type,
                "journey_date": str(self.date.date()), "journey_id": self.jid}


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Replay real journeys as a live feed and send predictions.")
    ap.add_argument("--date", default="2026-08-20", help="journey date to replay (use the test month)")
    ap.add_argument("--trains", type=int, default=30, help="how many random trains")
    ap.add_argument("--train-list", nargs="*", default=None, help="specific train numbers instead of random")
    ap.add_argument("--min-stops", type=int, default=6, help="only journeys with at least this many stops")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--speed", type=float, default=60.0, help="simulated seconds per real second")
    ap.add_argument("--flush", type=float, default=2.0, help="send a batch every N real seconds")
    ap.add_argument("--start-time", default=None, help="replay from this time of day, e.g. 07:00")
    ap.add_argument("--hours", type=float, default=None, help="replay this many hours (with --start-time)")
    ap.add_argument("--url", default="http://localhost:3000/api/predictions")
    ap.add_argument("--dry-run", action="store_true", help="do not contact the backend")
    ap.add_argument("--fast", action="store_true", help="no waiting (for testing / generating examples)")
    ap.add_argument("--features", default="processed/features/year=2026")
    ap.add_argument("--clean", default="processed/clean_data/year=2026", help="used only for train names")
    ap.add_argument("--models", default="models/multistop")
    ap.add_argument("--log", default="sim_log.jsonl")
    ap.add_argument("--outbox", default="sim_outbox.jsonl")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    meta, models = load_models(args.models)
    ctx = {"models": models, "cats": meta["categories"], "live_feats": meta["live"],
           "pre_feats": meta["pre_departure"], "scale": meta.get("range_scale", {}),
           "blend": meta.get("blend")}
    day = pd.Timestamp(args.date)

    # ---- choose journeys
    jl = find_journeys(args.features, day, args.threads)
    if jl.empty:
        sys.exit(f"No journeys found on {day.date()}.")
    jl["segments"] = jl["segments"].astype(int)
    jl["cover"] = jl["cover"].astype(float)
    if args.train_list:
        pick = jl[jl["train_no"].isin([str(t) for t in args.train_list])]
        missing = sorted(set(map(str, args.train_list)) - set(pick["train_no"]))
        if missing:
            print(f"  not running on {day.date()}: {', '.join(missing)}")
    else:
        pool = jl[(jl["segments"] >= args.min_stops - 1) & (jl["cover"] >= 0.8)]
        pick = pool.sample(min(args.trains, len(pool)), random_state=args.seed)
    if pick.empty:
        sys.exit("No journeys selected.")

    journeys = []
    for jid, r in enumerate(pick.itertuples(), 1):
        df = pd.read_parquet(os.path.join(args.features, r.file))
        J = df[(df["date"] == day) & (df["train_no"].astype(str) == r.train_no)].sort_values("seg_idx")
        name = None
        cpath = os.path.join(args.clean, r.file)
        if os.path.exists(cpath):
            try:
                nm = pd.read_parquet(cpath, columns=["train_name"])["train_name"].dropna()
                name = str(nm.iloc[0]) if len(nm) else None
            except Exception:
                pass
        journeys.append(Journey(jid, J, name, ctx))
    print(f"Replaying {len(journeys)} journeys from {day.date()}  |  speed {args.speed:g}x  |  "
          f"batch every {args.flush:g}s  |  {'DRY RUN' if args.dry_run else args.url}")

    # ---- event stream (pre-departure forecasts 30 simulated minutes before each start)
    events = []
    for j in journeys:
        events.append((add_min(j.first_time(), -30), -1, j.jid, -1, "pre_departure", None))
        events.extend(j.events())
    events.sort(key=lambda e: (e[0], e[2], e[3], e[1]))
    by_id = {j.jid: j for j in journeys}
    win_start = win_end = None
    if args.start_time:
        hh, mm = map(int, args.start_time.split(":"))
        win_start = add_min(day, hh * 60 + mm)
        if args.hours:
            win_end = add_min(win_start, args.hours * 60)
    t_first = events[0][0] if win_start is None else win_start
    t_last = events[-1][0] if win_end is None else win_end

    sender = Sender(args.url, args.dry_run, args.log, args.outbox)
    sender.send({"type": "simulation_start", "sent_at": iso(pd.Timestamp.now()),
                 "settings": {"date": str(day.date()), "speed": args.speed, "flush_seconds": args.flush,
                              "sim_start": iso(t_first), "sim_end": iso(t_last),
                              "note": "Replay of real journeys from the test month; predictions use only "
                                      "information available at each simulated moment."},
                 "journeys": [dict(j.train_info(), stops=j.stations()) for j in journeys]})

    stats = {"checks": 0, "abs_err_sum": 0.0, "in_range": 0, "updates": 0}

    def stat_block():
        c = stats["checks"]
        return {"updates_sent": stats["updates"], "checks": c,
                "mean_abs_error_min": round(stats["abs_err_sum"] / c, 2) if c else None,
                "within_range_pct": round(100 * stats["in_range"] / c, 1) if c else None}

    def handle(ev, sim_now):
        t, _, jid, i, kind, delay = ev
        j = by_id[jid]
        upd = {"type": "prediction_update", "sim_time": iso(sim_now), "train": j.train_info()}
        if kind == "pre_departure":
            rows = np.arange(j.n)
            p50, lo, hi = j._pre(rows)
            upd.update(trigger={"event": "pre_departure", "station": str(j.J["from_stn"].iloc[0]),
                                "scheduled": iso(j.J["sch_dep_A"].iloc[0])},
                       mode="pre_departure", current_delay_min=None,
                       predictions=j.prediction_rows(rows, p50, lo, hi, None, p50, sim_now, "pre_departure"),
                       completed_stops=[], check=None)
            return upd
        J = j.J
        check = None
        if kind == "arrival":           # train reached the station at the end of row i
            station, sch = str(J["to_stn"].iloc[i]), J["sch_arr_B"].iloc[i]
            prev = j.last_pred.pop(int(i), None)
            if prev is not None:
                b, lo, hi, made_at, mode = prev
                err = b - delay
                inr = lo <= delay <= hi
                stats["checks"] += 1
                stats["abs_err_sum"] += abs(err)
                stats["in_range"] += int(inr)
                check = {"station": station, "stop_no": int(i) + 1, "predicted_delay_min": num(b),
                         "range_low_min": num(lo), "range_high_min": num(hi), "actual_delay_min": num(delay),
                         "error_min": num(err), "was_in_range": bool(inr), "predicted_at": iso(made_at),
                         "prediction_mode": mode}
            j.completed.append({"stop_no": int(i) + 1, "station": station, "scheduled_arrival": iso(sch),
                                "actual_arrival": iso(plus_min(sch, delay)), "delay_min": num(delay)})
            nxt = i + 1                 # predict onward from this station (arrival known, departure not yet)
            res = j.predict_live(nxt, "arrival", delay) if nxt < j.n else None
        else:                           # departed from the station at the start of row i
            station, sch = str(J["from_stn"].iloc[i]), J["sch_dep_A"].iloc[i]
            if i == 0:
                j.completed.append({"stop_no": 0, "station": station, "scheduled_departure": iso(sch),
                                    "actual_departure": iso(plus_min(sch, delay)), "delay_min": num(delay)})
            res = j.predict_live(i, "departure")
        trig = {"event": kind, "station": station, "stop_no": int(i) + (1 if kind == "arrival" else 0),
                "scheduled": iso(sch), "actual": iso(plus_min(sch, delay)), "delay_min": num(delay)}
        if res is None:
            preds, mode, cur = [], ("finished" if kind == "arrival" and i + 1 >= j.n else "no_live_data"), num(delay)
        else:
            k, ahead, b50, blo, bhi, live50, pre50 = res
            preds = j.prediction_rows(ahead, b50, blo, bhi, live50, pre50, sim_now, "live")
            mode, cur = "live", num(k)
        j.current_delay = cur
        upd.update(trigger=trig, mode=mode, current_delay_min=cur, predictions=preds,
                   completed_stops=list(j.completed), check=check)
        return upd

    # ---- replay loop
    buffer, idx, batch_no = [], 0, 0
    sim_now = t_first

    def flush(sim_now):
        nonlocal buffer, batch_no
        if not buffer:
            return
        batch_no += 1
        stats["updates"] += len(buffer)
        sender.send({"type": "batch", "batch_no": batch_no, "sent_at": iso(pd.Timestamp.now()),
                     "sim_time": iso(sim_now), "updates": buffer, "stats": stat_block()})
        buffer = []

    if win_start is not None:
        # events before the window only build up each train's state; then every train that is
        # already running (or about to start) gets one snapshot at the start of the window
        latest = {}
        while idx < len(events) and events[idx][0] < win_start:
            upd = handle(events[idx], win_start)
            latest[events[idx][2]] = upd
            idx += 1
        stats.update(checks=0, abs_err_sum=0.0, in_range=0)              # score only the window
        buffer.extend(u for u in latest.values() if u["mode"] != "finished")
    real_min = (t_last - t_first).total_seconds() / args.speed / 60
    print(f"Simulated time {pd.Timestamp(t_first):%d %b %H:%M} -> {pd.Timestamp(t_last):%d %b %H:%M}"
          + ("" if args.fast else f"  (about {real_min:.0f} real minutes at {args.speed:g}x; Ctrl+C stops)"))
    wall0 = last_flush = time.monotonic()
    try:
        while idx < len(events):
            if args.fast:
                sim_now = events[idx][0]
            else:
                sim_now = add_min(t_first, (time.monotonic() - wall0) * args.speed / 60)
            if win_end is not None and sim_now > win_end:
                break
            while idx < len(events) and events[idx][0] <= sim_now:
                if win_end is not None and events[idx][0] > win_end:
                    break
                buffer.append(handle(events[idx], events[idx][0]))
                idx += 1
            if args.fast:
                if len(buffer) >= 50:
                    flush(sim_now)
                continue
            if time.monotonic() - last_flush >= args.flush:
                flush(sim_now)
                last_flush = time.monotonic()
                if batch_no % 15 == 0:
                    s_ = stat_block()
                    print(f"  sim {pd.Timestamp(sim_now):%d %b %H:%M}  events {idx:,}/{len(events):,}  "
                          f"checks {s_['checks']}  MAE {s_['mean_abs_error_min']}  in range {s_['within_range_pct']}%")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    flush(sim_now)
    s = stat_block()
    sender.send({"type": "simulation_end", "sent_at": iso(pd.Timestamp.now()), "sim_time": iso(sim_now),
                 "stats": s})
    sender.close()
    print(f"Done. {s['updates_sent']} updates, {s['checks']} prediction checks, MAE {s['mean_abs_error_min']} min, "
          f"{s['within_range_pct']}% inside the range.")
    print(f"All messages logged to {args.log}"
          + ("" if args.dry_run else f"; delivered {sender.sent}, failed {sender.failed}"
             + (f" (saved in {args.outbox})" if sender.failed else "")))


if __name__ == "__main__":
    main()
