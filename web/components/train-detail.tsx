"use client";

import Link from "next/link";
import { useState } from "react";
import { ChartNoAxesCombined, List, TrainFront } from "lucide-react";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { clock, day, delayTone, interval, minutes, Status } from "@/lib/display";
import type { Forecast } from "@/lib/replay-state";
import { useReplay } from "./shell";

function ForecastChart({ history }: { history: Forecast[] }) {
  const data = history.map((item) => ({
    time: Date.parse(item.sim_time),
    timestamp: item.sim_time,
    best: item.predicted_delay_min,
    range: item.range_low_min == null || item.range_high_min == null ? null : [item.range_low_min, item.range_high_min],
  }));
  if (!data.some((entry) => entry.best != null)) return <p className="chart-empty">No forecasts have been issued for this stop yet.</p>;
  const start = data[0].time;
  const end = data.at(-1)!.time;
  const formatter = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Kolkata", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  });
  return (
    <>
      <div className="chart" aria-label="Forecast delay and range over simulation time">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 14, right: 22, bottom: 9, left: 3 }}>
            <CartesianGrid vertical={false} stroke="#e7e1d8" />
            <XAxis dataKey="time" type="number" scale="time" domain={start === end ? [start - 300_000, end + 300_000] : [start, end]}
              fontSize={12} tickFormatter={(value) => formatter.format(value)} minTickGap={30} />
            <YAxis unit="m" fontSize={12} width={46} />
            <Tooltip labelFormatter={(value) => formatter.format(Number(value))}
              formatter={(value, name) => [
                Array.isArray(value) ? interval(Number(value[0]), Number(value[1])) : minutes(typeof value === "number" ? value : null),
                name,
              ]} />
            <Area type="linear" dataKey="range" name="Range" stroke="none" fill="#e4ac92" fillOpacity={0.48} isAnimationActive={false} />
            <Line type="linear" dataKey="best" name="Central forecast" stroke="#ae3c1f" strokeWidth={2.5}
              dot={{ r: 3, fill: "#ae3c1f" }} activeDot={{ r: 5 }} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <details className="history-details"><summary>Forecast history table</summary>
        <div className="table-scroll"><table><thead><tr><th scope="col">Issued (IST)</th><th scope="col">Source</th><th scope="col">Central forecast</th><th scope="col">Range</th></tr></thead>
          <tbody>{history.map((entry, index) => (
            <tr key={index}><td>{day(entry.sim_time)} · {clock(entry.sim_time)}</td>
              <td>{entry.mode === "live" ? "Live" : "Pre-departure"}</td>
              <td>{minutes(entry.predicted_delay_min)}</td><td>{interval(entry.range_low_min, entry.range_high_min)}</td></tr>
          ))}</tbody></table></div>
      </details>
    </>
  );
}

export function TrainDetail({ id }: { id: string }) {
  const { state, connected } = useReplay();
  const journey = state.start?.journeys.find((item) => String(item.journey_id) === id);
  const [chosen, setChosen] = useState<{ run: string | undefined; stop: number } | null>(null);

  if (!journey) {
    return (
      <section className="panel empty-state detail-empty">
        <TrainFront size={26} aria-hidden="true" />
        <h1>{connected ? "Journey unavailable" : "Loading journey"}</h1>
        <p>{connected ? "This journey ID is outside the current replay." : "Connecting to the replay stream."}</p>
        <Link className="text-button" href="/">Back to dashboard</Link>
      </section>
    );
  }

  const progress = state.progress[journey.journey_id];
  const latest = progress?.latest;
  const completed = new Map(latest?.completed_stops.map((stop) => [stop.stop_no, stop]));
  const firstForecastStop = latest?.predictions[0]?.stop_no;
  const stopNo = (chosen && chosen.run === state.start?.sent_at ? chosen.stop : undefined)
    ?? firstForecastStop ?? journey.stops.at(-1)!.stop_no;
  const target = journey.stops.find((stop) => stop.stop_no === stopNo);

  return (
    <>
      <div className="breadcrumbs"><Link href="/">Live dashboard</Link><span>/</span><span>{journey.train_no}</span></div>
      <section className="hero detail-hero">
        <div>
          <span className="eyebrow"><TrainFront size={15} aria-hidden="true" /> Train {journey.train_no} · {journey.train_type}</span>
          <h1>{journey.train_name ?? `Train ${journey.train_no}`}</h1>
          <p>{journey.stops[0].station} to {journey.stops.at(-1)?.station} · {day(journey.journey_date)} · {journey.stops.length} stops</p>
        </div>
        <Status mode={latest?.mode} />
      </section>

      <section className="panel chart-panel" aria-labelledby="history-heading">
        <div className="panel-title">
          <div className="title-icon"><ChartNoAxesCombined size={19} aria-hidden="true" /></div>
          <div><h2 id="history-heading">Forecast history</h2><p>Central delay and forecast range by issue time</p></div>
          <label className="chart-picker"><span>Stop</span><select value={stopNo}
            onChange={(event) => setChosen({ run: state.start?.sent_at, stop: Number(event.target.value) })}>
            {journey.stops.slice(1).map((stop) => <option key={stop.stop_no} value={stop.stop_no}>{stop.station} · stop {stop.stop_no}</option>)}
          </select></label>
        </div>
        <ForecastChart key={target?.stop_no} history={progress?.history[stopNo] ?? []} />
      </section>

      <section className="panel timeline" aria-labelledby="timeline-heading">
        <div className="panel-title"><div className="title-icon"><List size={19} aria-hidden="true" /></div>
          <div><h2 id="timeline-heading">Stop timeline</h2><p>Latest forecast shown for each stop</p></div>
        </div>
        <div className="table-scroll">
          <table>
            <thead><tr><th scope="col">Station</th><th scope="col">Scheduled</th><th scope="col">Delay (BEST / PRE)</th>
              <th scope="col">Forecast range</th><th scope="col">Expected arrival</th><th scope="col">Actual delay</th></tr></thead>
            <tbody>{journey.stops.map((stop) => {
              const forecast = progress?.history[stop.stop_no]?.at(-1);
              const actual = completed.get(stop.stop_no);
              const actualTime = actual && ("actual_arrival" in actual ? actual.actual_arrival : actual.actual_departure);
              const scheduled = stop.stop_no === 0 ? stop.scheduled_departure : stop.scheduled_arrival;
              return (
                <tr key={stop.stop_no} className={stop.stop_no === stopNo ? "chosen-row" : undefined}>
                  <th scope="row">
                    {stop.stop_no === 0 ? stop.station : <button className="text-button" type="button"
                      aria-pressed={stop.stop_no === stopNo} aria-label={`Show forecast history for ${stop.station}, stop ${stop.stop_no}`}
                      onClick={() => setChosen({ run: state.start?.sent_at, stop: stop.stop_no })}>{stop.station}</button>}
                    <small>{stop.stop_no === 0 ? "Origin departure" : `Stop ${stop.stop_no}`}</small>
                  </th>
                  <td>{clock(scheduled)}{scheduled && <small>{day(scheduled)}</small>}</td>
                  <td className={delayTone(forecast?.predicted_delay_min)}>{minutes(forecast?.predicted_delay_min)}
                    {forecast && <small>{forecast.mode === "live" ? "BEST" : "PRE"}</small>}</td>
                  <td>{interval(forecast?.range_low_min, forecast?.range_high_min)}</td>
                  <td>{clock(forecast?.expected_arrival)}{forecast?.expected_arrival && <small>{day(forecast.expected_arrival)}</small>}</td>
                  <td className={delayTone(actual?.delay_min)}>{minutes(actual?.delay_min)}{actualTime && <small>{clock(actualTime)} actual</small>}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      </section>
    </>
  );
}
