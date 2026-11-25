"""Read the prepared station records for the route finder. Run from any directory."""
import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq


WEB = Path(__file__).resolve().parents[1]
PROCESSED = WEB.parent / "processed"
CLEAN = PROCESSED / "clean_data" / "year=2026"
CATALOG = WEB / "data" / "train-catalog.json"


def catalog_entry(entry):
    file = entry["file"]
    batch = next(pq.ParquetFile(CLEAN / file).iter_batches(batch_size=1, columns=["train_name"]))
    return [entry["train_no"], batch.column(0)[0].as_py(), file]


def build_catalog():
    with (PROCESSED / "features/year=2026/_train_index.csv").open() as source:
        with ThreadPoolExecutor(max_workers=8) as pool:
            entries = list(pool.map(catalog_entry, csv.DictReader(source)))
    CATALOG.parent.mkdir(exist_ok=True)
    CATALOG.write_text(json.dumps(entries, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Indexed {len(entries)} trains")


def finite(value):
    return value if value is not None and math.isfinite(value) else None


def location(lat, lng):
    if finite(lat) is not None and finite(lng) is not None and abs(lat) <= 90 and abs(lng) <= 180 and (lat or lng):
        return [lat, lng]
    return None


def stamp(value):
    return value.isoformat(timespec="seconds") + "+05:30" if value is not None else None


def journey(file, date):
    columns = ["date", "stop_idx", "train_type", "station_code", "platform", "sch_arr",
               "sch_dep", "act_arr", "delay_arr_min", "delay_dep_min", "distance_km", "gps_lat", "gps_lng"]
    records = pq.ParquetFile(CLEAN / file).read(columns=columns).to_pylist()
    dates = sorted({r["date"].date().isoformat() for r in records}, reverse=True)
    chosen = date or dates[0]
    rows = sorted((r for r in records if r["date"].date().isoformat() == chosen), key=lambda r: r["stop_idx"])
    if not rows:
        return None
    stations = pq.ParquetFile(CLEAN / "_stations.parquet").read().to_pylist()
    coordinates = {s["station_code"]: location(s["stn_lat"], s["stn_lng"]) for s in stations}
    stops = [{
        "index": r["stop_idx"], "station": r["station_code"], "platform": r["platform"],
        "arrival": stamp(r["sch_arr"]), "departure": stamp(r["sch_dep"]),
        "actual": stamp(r["act_arr"]), "delay": finite(r["delay_arr_min"]),
        "hasObservedDelay": any(finite(r[c]) is not None for c in ("delay_arr_min", "delay_dep_min")),
        "km": finite(r["distance_km"]),
        "point": coordinates.get(r["station_code"]) or location(r["gps_lat"], r["gps_lng"]),
    } for r in rows]
    return {"date": chosen, "dates": dates, "type": rows[0]["train_type"], "stops": stops}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog")
    route = commands.add_parser("route")
    route.add_argument("file", type=lambda s: Path(s).name)
    route.add_argument("date", nargs="?")
    args = parser.parse_args()
    if args.command == "catalog":
        build_catalog()
    else:
        print(json.dumps(journey(args.file, args.date), allow_nan=False))
