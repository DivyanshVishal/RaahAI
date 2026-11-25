"use client";

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useEffect, useState, useTransition } from "react";
import { CalendarDays, MapPinned } from "lucide-react";
import { clock, day, interval, minutes } from "@/lib/display";
import type { ManualRow } from "@/lib/manual-prediction";
import type { JourneyRoute, Train } from "@/lib/train-routes";
import { useLiveFeed } from "@/lib/use-live-feed";

const RouteMap = dynamic(() => import("./route-map"), {
  ssr: false, loading: () => <div className="atlas-map map-placeholder">Loading route map…</div>,
});
function time(value: string | null, date: string) {
  if (!value) return "—";
  const offset = (Date.parse(value.slice(0, 10)) - Date.parse(date)) / 86400000;
  return <time dateTime={value} title={day(value)}>{clock(value)}{offset !== 0 && <sup>{offset > 0 ? "+" : ""}{offset}d</sup>}</time>;
}

// Both sources use local railway time. The CSV uses a space where the route uses T.
function arrivalKey(station: string, schedule: string) {
  return station + "|" + schedule.slice(0, 19).replace(" ", "T");
}

export function RouteDetails({ train, route }: { train: Pick<Train, "number" | "name">; route: JourneyRoute }) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [selected, setSelected] = useState<number | null>(null);
  const { stops, date } = route;
  const [from, setFrom] = useState("");
  const [automatic, setAutomatic] = useState(false);
  const [replayNumber, setReplayNumber] = useState(0);
  const canUseFeed = date >= "2026-07-01" && date < "2026-09-01";
  const feed = useLiveFeed(automatic, train.number, date, replayNumber);
  const lastReport = feed.events.at(-1);
  const keyFor = (row: { stop_index: number; station: string }) => row.stop_index + "|" + row.station;
  const arrivals = new Map(feed.events.filter((event) => event.type === "arrival").map((event) => [keyFor(event), event]));
  const liveRows = new Map(feed.batch?.forecasts?.map((forecast) => [keyFor(forecast), forecast]));
  const scores = feed.batch?.scores ?? [];
  const meanError = (field: "pre" | "delay") =>
    (scores.reduce((sum, row) => sum + Math.abs(row[field] - row.actual), 0) / scores.length).toFixed(1);
  const stationChoices = stops.slice(0, -1).filter((s) =>
    s.hasObservedDelay && stops.filter((other) => other.station === s.station).length === 1
  );
  const [forecasts, setForecasts] = useState<Map<string, ManualRow | null> | null>(null);
  const [baseline, setBaseline] = useState<Map<string, ManualRow | null> | null>(null);
  const [forecastError, setForecastError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    async function loadForecasts() {
      try {
        const response = await fetch("/api/predict", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ train: train.number, date, station: from || undefined }), signal: controller.signal,
        });
        if (!response.ok) throw new Error("Prediction unavailable");
        const { rows }: { rows: ManualRow[] } = await response.json();
        const byArrival = new Map<string, ManualRow | null>();
        for (const row of rows) {
          if (!row.sch_arrival) continue;
          const key = arrivalKey(row.station, row.sch_arrival);
          byArrival.set(key, byArrival.has(key) ? null : row);
        }
        if (!controller.signal.aborted) {
          setForecasts(byArrival);
          if (!from) setBaseline(byArrival);
        }
      } catch {
        if (!controller.signal.aborted) setForecastError(true);
      }
    }
    void loadForecasts();
    return () => controller.abort();
  }, [train.number, date, from]);
  const first = stops[0], last = stops[stops.length - 1];
  const departure = first.departure || first.arrival, arrival = last.arrival || last.departure;
  const duration = departure && arrival ? Math.round((Date.parse(arrival) - Date.parse(departure)) / 60000) : null;
  const distance = first.km !== null && last.km !== null ? last.km - first.km : null;
  const mapped = stops.filter((s) => s.point).length;
  return <>
    <section className="atlas-grid" aria-label="Journey overview">
      <div className="journey-sheet">
        <span className="train-no">Train {train.number}</span>
        <h2>{train.name || "Unnamed train"}</h2>
        <label className="journey-date"><span><CalendarDays size={15} />Recorded journey</span>
          <select value={date} disabled={pending} onChange={(e) => {
            const url = "/?" + new URLSearchParams({ train: train.number, date: e.target.value });
            startTransition(() => router.push(url, { scroll: false }));
          }}>{route.dates.map((d) => <option key={d} value={d}>{day(d)}</option>)}</select>
        </label>
        <div className="route-ends">
          <div><i /><span>First recorded stop<strong>{first.station}</strong></span><b>{time(departure, date)}</b></div>
          <div><i /><span>Last recorded stop<strong>{last.station}</strong></span><b>{time(arrival, date)}</b></div>
        </div>
        <dl className="route-stats">
          <div><dt>Stops</dt><dd>{stops.length}</dd></div>
          {distance !== null && distance >= 0 && <div><dt>Distance</dt><dd>{Number(distance.toFixed(1))} km</dd></div>}
          {duration !== null && duration >= 0 && <div><dt>Scheduled time</dt><dd>{Math.floor(duration / 60)}h {duration % 60}m</dd></div>}
        </dl>
      </div>
      <section className="route-map-panel">
        <header><h2><MapPinned size={18} /> Route map</h2>{selected !== null && <button onClick={() => setSelected(null)}>Show full route</button>}</header>
        <RouteMap stops={stops} selected={selected} onSelect={setSelected} />
        <p>{mapped} of {stops.length} stops mapped. The line connects stations in journey order.</p>
      </section>
    </section>
    <section className="station-panel">
      <header><div><h2>Station timetable</h2><p>{automatic
        ? "Forecasts update as recorded station reports arrive."
        : from
        ? "Updated forecasts use the delay recorded at " + from + " and cover subsequent stops."
        : "Choose an earlier stop to update the forecast using its recorded delay."} Times are in IST.</p>
        <p role="status">{automatic ? feed.message : forecastError ? "Forecast unavailable for this selection." : forecasts === null ? "Loading delay forecasts…" : ""}</p>
        {automatic && feed.batch && <p>Simulated time {time(feed.batch.simulated_at, date)} IST · {feed.batch.speed}×
          {!!feed.batch.max_gap_seconds && <> base speed · {feed.batch.max_gap_seconds}s max gap</>} · {feed.events.length} station reports
          {lastReport && <> · Latest {lastReport.station} {lastReport.type}</>}</p>}
        {automatic && scores.length > 0 && <p>MAE across {scores.length} scored arrivals · Before departure {meanError("pre")} min · Updated {meanError("delay")} min</p>}</div>
        <div className="atlas-forecast-controls">
        {!automatic && <label className="journey-date">Forecast from
          <select value={from} onChange={(event) => {
            setForecasts(null);
            setForecastError(false);
            setFrom(event.target.value);
          }}>
            <option value="">Before departure</option>
            {stationChoices.map((s) => <option key={s.station} value={s.station}>{s.station}</option>)}
          </select>
        </label>}
        {canUseFeed && <button type="button" aria-pressed={automatic} onClick={() => {
          if (!automatic) setFrom("");
          setAutomatic(!automatic);
        }}>{automatic ? "Show historical timetable" : "Connect simulated live feed"}</button>}
        {automatic && feed.batch && <button type="button" onClick={() => setReplayNumber((run) => run + 1)}>Restart simulation</button>}
        </div></header>
      <div className="station-scroll" tabIndex={0} role="region" aria-label="Scrollable station timetable">
        <table><thead><tr>{["Stop", "Station", "Arrival", "Departure", "Before departure", ...(from || automatic ? ["Updated delay"] : []), "Predicted range", "Recorded delay", "Recorded arrival", "Platform", "Distance"].map((s) => <th key={s} scope="col">{s}</th>)}</tr></thead>
          <tbody>{stops.map((s, i) => {
            const key = i > 0 && s.arrival ? arrivalKey(s.station, s.arrival) : "";
            const forecast = forecasts?.get(key);
            const pre = baseline?.get(key) ?? forecast;
            const live = liveRows.get(s.index + "|" + s.station);
            const observed = arrivals.get(s.index + "|" + s.station);
            const actual = automatic ? observed?.actual_at ?? null : s.actual;
            const recorded = automatic ? observed?.delay_min ?? null : s.delay;
            const updated = automatic ? live?.delay : forecast?.best_delay;
            const low = automatic ? live?.low ?? pre?.pre_low : from ? forecast?.best_low : pre?.pre_low;
            const high = automatic ? live?.high ?? pre?.pre_high : from ? forecast?.best_high : pre?.pre_high;
            return <tr key={i} className={selected === i ? "station-selected" : ""}>
            <td>{String(i + 1).padStart(2, "0")}</td><th scope="row">{s.point ? <button title={"Locate " + s.station + " on the map"} aria-pressed={selected === i} onClick={() => {
              setSelected(i);
              document.querySelector(".route-map-panel")?.scrollIntoView({ block: "nearest" });
            }}>{s.station}</button> : s.station}</th>
            <td>{time(s.arrival, date)}</td><td>{time(s.departure, date)}</td>
            <td>{minutes(pre?.pre_delay)}</td>{(from || automatic) && <td>
              {minutes(updated)}{automatic && live && <small>From {live.from_station} {live.from_type}</small>}</td>}
            <td>{interval(low, high)}</td>
            <td className={recorded !== null && recorded > 0 ? "late-stop" : ""}>{minutes(recorded)}</td>
            <td>{time(actual, date)}</td>
            <td>{s.platform || "—"}</td><td>{s.km !== null ? Number(s.km.toFixed(1)) + " km" : "—"}</td>
          </tr>; })}</tbody></table>
      </div>
    </section>
  </>;
}
