"""Serve a shared, clock-paced feed of recorded Indian Railways observations."""

import argparse
from bisect import bisect_right
from collections import OrderedDict
import csv
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import accumulate, pairwise
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "processed"
IST = timezone(timedelta(hours=5, minutes=30))
DEFAULT_MAX_GAP = 5.0
# train_multistop.py splits on journey start date using half-open intervals.
SPLITS = {
    "validation": (date(2026, 7, 1), date(2026, 8, 1)),
    "test": (date(2026, 8, 1), date(2026, 9, 1)),
}


def stamp(value):
    return value.replace(tzinfo=IST).isoformat() if value is not None else None


def train_partition(number):
    with (DATA / "features/year=2026/_train_index.csv").open() as index:
        return next((r["file"] for r in csv.DictReader(index) if r["train_no"] == number), None)


def load_events(train, day, split, partition=None):
    start, end = SPLITS[split]
    if not start <= day < end:
        raise ValueError(f"{split} requires a journey date from {start} through {end - timedelta(days=1)}")
    if not re.fullmatch(r"[0-9]{1,5}", train):
        raise ValueError("Train number must contain one to five digits")
    number = str(int(train))
    partition = partition or train_partition(number)
    if partition is None:
        raise ValueError(f"Unknown train {train}")
    columns = ["station_code", "stop_idx", "reported_at", "reported", "departed",
               "act_arr", "sch_arr", "delay_arr_min", "act_dep", "sch_dep", "delay_dep_min"]
    rows = pq.read_table(DATA / "clean_data/year=2026" / partition, columns=columns,
                         filters=[("date", "=", datetime.combine(day, datetime.min.time())),
                                  ("train_no", "=", number)]).to_pylist()
    events = []
    for row in rows:
        reported = row["reported_at"]
        if reported is None:
            continue
        for kind, suffix, flag in [("arrival", "arr", "reported"), ("departure", "dep", "departed")]:
            actual = row["act_" + suffix]
            if actual is None or not row[flag]:
                continue
            available = max(actual, reported)
            delay = row["delay_" + suffix + "_min"]
            events.append((available, actual, row["stop_idx"], kind, {
                "type": kind, "station": row["station_code"], "stop_index": row["stop_idx"],
                "scheduled_at": stamp(row["sch_" + suffix]), "actual_at": stamp(actual),
                "available_at": stamp(available), "delay_min": delay if delay is not None and math.isfinite(delay) else None,
            }))
    if not events:
        raise ValueError(f"No timestamped observations for train {train} on {day}")
    events.sort(key=lambda item: item[:4])
    first = events[0][0]
    return [((at - first).total_seconds(), dict(payload, id=i))
            for i, (at, _, _, _, payload) in enumerate(events, 1)]


class Feed:
    def __init__(self, events, speed, metadata, clock=time.monotonic, *, max_gap=DEFAULT_MAX_GAP):
        self.offsets = [offset for offset, _ in events]
        self.events = [event for _, event in events]
        self.speed = speed
        self.max_gap = max_gap
        gaps = ((right - left) / speed for left, right in pairwise(self.offsets))
        if max_gap:
            gaps = (min(gap, max_gap) for gap in gaps)
        self.release_times = list(accumulate(gaps, initial=0.0))
        self.metadata = metadata
        self.clock = clock
        self.started = None
        self.lock = threading.Lock()
        self.frames = None

    def poll(self, after):
        with self.lock:
            now = self.clock()
            if self.started is None:
                self.started = now
            elapsed = now - self.started
            count = bisect_right(self.release_times, elapsed)
        if after > count:
            raise ValueError("Cursor exceeds published events. Start with after=0.")
        start = datetime.fromisoformat(self.events[0]["available_at"])
        simulated_offset = self.offsets[count - 1]
        if count < len(self.events):
            # Interpolate the original clock across the compressed gap.
            lower, upper = self.release_times[count - 1:count + 1]
            fraction = (elapsed - lower) / (upper - lower)
            simulated_offset += fraction * (self.offsets[count] - simulated_offset)
        simulated_at = start + timedelta(seconds=simulated_offset)
        batch = {**self.metadata, "speed": self.speed, "cursor": count, "complete": count == len(self.events),
                 "max_gap_seconds": self.max_gap,
                 "simulated_at": simulated_at.isoformat(timespec="seconds"),
                 "events": self.events[after:count]}
        if self.frames is not None:
            batch.update(self.frames[count - 1])
        return batch


def forecast_frames(events, model):
    """Build snapshots in release order and score only forecasts issued before actual arrival."""
    latest = {}
    history = {}
    arrivals = set()
    scored = {}
    frames = []
    for _, event in events:
        if event["type"] == "arrival":
            index = event["stop_index"]
            arrivals.add(index)
            actual = datetime.fromisoformat(event["actual_at"])
            candidates = history.get(index, [])
            forecast = next((row for row in reversed(candidates)
                             if datetime.fromisoformat(row["available_at"]) < actual), None)
            if forecast is not None and event["delay_min"] is not None:
                scored[index] = dict(forecast, actual=event["delay_min"])
                latest[index] = forecast
            else:
                latest.pop(index, None)
        for forecast in model.update(event):
            index = forecast["stop_index"]
            if index not in arrivals:
                history.setdefault(index, []).append(forecast)
                latest[index] = forecast
        frames.append({"forecasts": list(latest.values()), "scores": list(scored.values())})
    return frames


class ReplaySessions:
    def __init__(self, speed, build_frames, *, max_gap=DEFAULT_MAX_GAP):
        self.speed = speed
        self.max_gap = max_gap
        self.build_frames = build_frames
        self.journeys = OrderedDict()
        self.sessions = OrderedDict()
        self.lock = threading.Lock()

    def open(self, train, day, split, session):
        key = (train, day, split)
        connection = (*key, session)
        with self.lock:
            if connection in self.sessions:
                self.sessions.move_to_end(connection)
                return self.sessions[connection]
            if key not in self.journeys:
                events = load_events(train, day, split)
                frames = self.build_frames(train, day, events) if self.build_frames else None
                self.journeys[key] = (events, frames)
                if len(self.journeys) > 16:
                    self.journeys.popitem(last=False)
            self.journeys.move_to_end(key)
            events, frames = self.journeys[key]
            feed = Feed(events, self.speed, {"train_no": train, "journey_date": day.isoformat(),
                                            "split": split, "run": uuid4().hex}, max_gap=self.max_gap)
            feed.frames = frames
            self.sessions[connection] = feed
            if len(self.sessions) > 32:
                self.sessions.popitem(last=False)
            return feed


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            self.reply(200, "text/html", (HERE / "index.html").read_bytes())
        elif url.path == "/api/events":
            self.serve_events({key: values[0] for key, values in parse_qs(url.query).items()})
        else:
            self.send_error(404)

    def serve_events(self, query):
        feed = self.server.feed
        try:
            after = int(query.get("after", "0"))
            if after < 0:
                raise ValueError("Cursor must be nonnegative")
            if "session" in query:
                session = query["session"]
                train = query.get("train_no", "")
                if not re.fullmatch(r"[a-f0-9-]{36}", session) or not re.fullmatch(r"[0-9]{1,5}", train):
                    raise ValueError("Choose a valid train and replay session.")
                day = date.fromisoformat(query.get("journey_date", ""))
                split = next((name for name, (start, end) in SPLITS.items() if start <= day < end), None)
                if split is None:
                    raise ValueError("Choose a recorded journey in July or August 2026.")
                feed = self.server.replays.open(str(int(train)), day, split, session)
            elif any(query.get(key, feed.metadata[key]) != feed.metadata[key] for key in ("train_no", "journey_date")):
                self.json_reply(404, {"error": "Feed is serving another journey."})
                return
            if query.get("run", feed.metadata["run"]) != feed.metadata["run"]:
                self.json_reply(409, {"error": "Feed restarted."})
                return
            self.json_reply(200, feed.poll(after))
        except ValueError as exc:
            self.json_reply(400, {"error": str(exc)})

    def json_reply(self, status, payload):
        self.reply(status, "application/json", json.dumps(payload, allow_nan=False).encode())

    def reply(self, status, mime, body):
        self.send_response(status)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--date", type=date.fromisoformat, help="Journey start date (default: July 21 for validation, August 20 for test)")
    parser.add_argument("--train", default="18636")
    parser.add_argument("--speed", type=float, default=120, help="Simulated seconds per wall-clock second")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP,
                        help="Maximum real seconds between reports (default: 5, use 0 for uncapped intervals)")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--observations-only", action="store_true", help="Serve station events without running model inference")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("Speed must be positive and finite")
    if not math.isfinite(args.max_gap) or args.max_gap < 0:
        parser.error("Maximum gap must be nonnegative and finite")
    day = args.date or (date(2026, 7, 21) if args.split == "validation" else date(2026, 8, 20))
    build_frames = None
    if not args.observations_only:
        sys.path.insert(0, str(DATA.parent))
        import pandas as pd
        from predict import load_models
        from live_feed.forecast import JourneyForecast

        settings, models = load_models(str(DATA.parent / "models/multistop"))

        def build_frames(number, journey_day, journey_events):
            partition = DATA / "features/year=2026" / train_partition(number)
            segments = pd.read_parquet(
                partition, filters=[("date", "=", pd.Timestamp(journey_day)), ("train_no", "=", number)],
            )
            if segments.empty:
                return None
            return forecast_frames(journey_events, JourneyForecast(segments, settings, models))

    replays = ReplaySessions(args.speed, build_frames, max_gap=args.max_gap)
    try:
        if not re.fullmatch(r"[0-9]{1,5}", args.train) or not SPLITS[args.split][0] <= day < SPLITS[args.split][1]:
            parser.error("Choose a valid train and journey date within the requested split.")
        feed = replays.open(str(int(args.train)), day, args.split, "default")
    except ValueError as exc:
        parser.error(str(exc))
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.feed = feed
        server.replays = replays
        print(f"Recorded train feed at http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
