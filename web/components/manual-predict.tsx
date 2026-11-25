"use client";

import { useState, type FormEvent } from "react";
import { FlaskConical, TrainFront } from "lucide-react";
import { clock, day, delayTone, interval, minutes } from "@/lib/display";
import type { ManualRow } from "@/lib/manual-prediction";
import { useReplay } from "./shell";

type Result = { mode: "live" | "pre_departure"; rows: ManualRow[] };

function estimatedArrival(schedule: string | null, delay: number | null | undefined) {
  if (!schedule || delay == null) return null;
  const parsed = Date.parse(`${schedule.replace(" ", "T")}+05:30`);
  const arrival = new Date(parsed + Math.round(delay * 60_000) + 330 * 60_000);
  if (!Number.isFinite(arrival.valueOf())) return null;
  return arrival.toISOString().slice(0, 19);
}

export function ManualPredict() {
  const { state } = useReplay();
  const [train, setTrain] = useState("");
  const [station, setStation] = useState("");
  const [delay, setDelay] = useState("");
  const [selectedDate, setSelectedDate] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<(Result & { train: string; date: string; station: string }) | null>(null);
  const journeys = state.start?.journeys ?? [];
  const date = selectedDate ?? state.start?.settings.date ?? "2026-08-20";
  const route = journeys.find((journey) => journey.train_no === train && journey.journey_date === date);

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          train, date, station: station.trim() || undefined,
          delay: station.trim() && delay !== "" ? Number(delay) : undefined,
        }),
      });
      const data = await response.json();
      if (!response.ok) setError(data.error);
      else setResult({ ...data, train, date, station: station.trim().toUpperCase() });
    } catch {
      setError("Unable to reach the local prediction service.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="hero">
        <div><span className="eyebrow"><FlaskConical size={15} aria-hidden="true" /> Manual forecast</span>
          <h1>What-if prediction</h1><p>Forecast arrival delays for a recorded journey using the Python models.</p></div>
      </section>
      <form className="panel predict-form" onSubmit={predict}>
        <div className="panel-title"><div className="title-icon"><TrainFront size={19} aria-hidden="true" /></div>
          <div><h2>Journey details</h2><p>Leave the current station blank for a pre-departure forecast.</p></div>
        </div>
        <div className="predict-fields">
          <label>Train number<input value={train} onChange={(event) => setTrain(event.target.value)} inputMode="numeric"
            pattern="[0-9]+" list="train-numbers" placeholder="e.g. 12189" required />
            <datalist id="train-numbers">{[...new Map(journeys.map((item) => [item.train_no, item])).values()].map((item) =>
              <option key={item.train_no} value={item.train_no}>{item.train_name}</option>)}</datalist>
          </label>
          <label>Journey date<input type="date" value={date} onChange={(event) => setSelectedDate(event.target.value)} required /></label>
          <label>Current station <span className="muted">(optional)</span><input value={station}
            onChange={(event) => setStation(event.target.value)} autoCapitalize="characters" placeholder="Station code" list="station-codes" />
            <datalist id="station-codes">{[...new Set(route?.stops.slice(0, -1).map((stop) => stop.station))].map((code) =>
              <option value={code} key={code} />)}</datalist>
          </label>
          <label>Current delay in minutes <span className="muted">(optional)</span><input value={delay}
            type="number" step="any" disabled={!station.trim()} onChange={(event) => setDelay(event.target.value)} placeholder="Use recorded delay" /></label>
        </div>
        <div className="predict-actions"><button className="button-primary" disabled={busy} type="submit">
          <FlaskConical size={16} aria-hidden="true" />{busy ? "Predicting…" : "Run prediction"}
        </button><p className="muted">Positive delay means late. Negative delay means early.</p></div>
        {busy && <p className="form-message" role="status">Running the Python models…</p>}
        {error && <p className="form-error" role="alert">{error}</p>}
      </form>
      {result && (
        <section className="panel results" aria-labelledby="results-heading">
          <div className="panel-title"><div className="title-icon"><TrainFront size={19} aria-hidden="true" /></div>
            <div><h2 id="results-heading">{result.mode === "live" ? "Live forecast" : "Pre-departure forecast"}</h2>
              <p>Train {result.train} · {day(result.date)}{result.station && ` · From ${result.station}`}</p></div>
          </div>
          <div className="table-scroll"><table>
            <thead><tr><th scope="col">Station</th><th scope="col">Scheduled</th>
              <th scope="col">{result.mode === "live" ? "BEST delay" : "PRE delay"}</th><th scope="col">Range</th>
              <th scope="col">Expected arrival</th>
              {result.mode === "live" && <><th scope="col">Live alone</th><th scope="col">PRE alone</th></>}
              <th scope="col">Recorded delay</th></tr></thead>
            <tbody>{result.rows.map((row) => {
              const isLive = result.mode === "live";
              const central = isLive ? row.best_delay : row.pre_delay;
              const eta = estimatedArrival(row.sch_arrival, central);
              return <tr key={row.stop}><th scope="row">{row.station}<small>Stop {row.stop}</small></th>
                <td>{clock(row.sch_arrival)}{row.sch_arrival && <small>{day(row.sch_arrival)}</small>}</td>
                <td className={delayTone(central)}>{minutes(central)}</td>
                <td>{interval(isLive ? row.best_low : row.pre_low, isLive ? row.best_high : row.pre_high)}</td>
                <td>{clock(eta)}{eta && <small>{day(eta)}</small>}</td>
                {isLive && <><td>{minutes(row.live_delay)}</td><td>{minutes(row.pre_delay)}</td></>}
                <td>{minutes(row.actual_delay)}</td></tr>;
            })}</tbody>
          </table></div>
          <p className="results-note">Recorded delays come from the original journey.</p>
        </section>
      )}
    </>
  );
}
