"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useState } from "react";
import { Clock3, MapPin, Radio, Search, TrainFront, Waypoints } from "lucide-react";
import type { Journey } from "@/lib/messages";
import { clock, day, delayTone, minutes, Status } from "@/lib/display";
import { useReplay } from "./shell";
import { Accuracy } from "./accuracy";
import { ReplayControls } from "./replay-controls";

const NetworkMap = dynamic(() => import("./network-map"), {
  ssr: false, loading: () => <div className="map-loading">Loading network map…</div>,
});
const noJourneys: Journey[] = [];

export function Dashboard() {
  const { state, connected } = useReplay();
  const [selection, setSelection] = useState<{ replay: string | undefined; id: number } | null>(null);
  const [query, setQuery] = useState("");
  const { start, time, progress } = state;
  const selected = selection && selection.replay === start?.sent_at ? selection.id : null;
  const journeys = start?.journeys ?? noJourneys;
  const active = journeys.filter((j) => {
    const mode = progress[j.journey_id]?.latest.mode;
    return mode === "live" || mode === "no_live_data";
  }).length;
  const filter = query.trim().toLowerCase();
  const visible = journeys.filter((journey) =>
    !filter || journey.train_no.includes(filter) || journey.train_name?.toLowerCase().includes(filter));
  const august = !start || start.settings.date.startsWith("2026-08-");

  return (
    <>
      <section className="hero">
        <div>
          <span className="eyebrow"><Radio size={15} aria-hidden="true" /> Journey replay</span>
          <h1>Live dashboard</h1>
          <p>{august ? "Replay of recorded August 2026 journeys. The model was never trained on August." :
            `Replay of recorded journeys from ${day(start.settings.date)}.`}</p>
        </div>
        <ReplayControls />
      </section>

      <div className="overview" aria-label="Replay overview">
        <div className="overview-item"><div className="overview-icon"><Clock3 size={19} aria-hidden="true" /></div>
          <div><p>Replay clock <span>IST</span></p><strong>{clock(time)}</strong>{time && <small>{day(time)}</small>}</div>
        </div>
        <div className="overview-item"><div className="overview-icon"><TrainFront size={19} aria-hidden="true" /></div>
          <div><p>Trains running</p><strong>{start ? active : "—"}</strong>{start && <small>of {journeys.length} journeys</small>}</div>
        </div>
        <div className="overview-item"><div className="overview-icon"><Radio size={19} aria-hidden="true" /></div>
          <div><p>Replay status</p><strong className="overview-status">{state.phase === "idle" ? "Ready to replay" : state.phase === "running" ? "In progress" : "Complete"}</strong>
            <small>{start ? `${start.settings.speed}× speed` : connected ? "Receiver online" : "Connecting to receiver"}</small></div>
        </div>
      </div>

      <div className="dashboard-grid">
        <section className="panel map-panel" aria-labelledby="map-heading">
          <div className="panel-title"><div className="title-icon"><MapPin size={19} aria-hidden="true" /></div>
            <div><h2 id="map-heading">Network map</h2><p>Stations and latest replayed train positions</p></div>
          </div>
          <NetworkMap journeys={journeys} progress={progress} selected={selected} />
          <div className="map-legend">
            <span><i /> Station</span><span><i className="train-dot" /> Train position</span>
            {selected !== null && <button className="text-button" onClick={() => setSelection(null)}>Show all routes</button>}
          </div>
        </section>
        <Accuracy state={state} />
      </div>

      <section className="panel fleet" aria-labelledby="fleet-heading">
        <div className="panel-title fleet-title">
          <div className="title-icon"><Waypoints size={19} aria-hidden="true" /></div>
          <div><h2 id="fleet-heading">Train journeys</h2><p>{start ? `${visible.length} of ${journeys.length} journeys` : "Awaiting a replay"}</p></div>
          {start && <label className="search"><Search size={16} aria-hidden="true" />
            <input aria-label="Search by train number or name" placeholder="Find a train" value={query} onChange={(e) => setQuery(e.target.value)} />
          </label>}
        </div>
        {visible.length > 0 ? (
          <div className="table-scroll">
            <table>
              <thead><tr><th scope="col">Train</th><th scope="col">Status</th><th scope="col">Current delay</th>
                <th scope="col">Next station</th><th scope="col">Expected arrival</th><th scope="col">Map</th></tr></thead>
              <tbody>
                {visible.map((journey) => {
                  const id = journey.journey_id;
                  const latest = progress[id]?.latest;
                  const next = latest?.predictions[0];
                  const anchor = latest?.mode === "no_live_data" && "stop_no" in latest.trigger ? latest.trigger.stop_no : null;
                  const nextStation = next?.station ?? (anchor == null ? undefined :
                    journey.stops.find((stop) => stop.stop_no === anchor + 1)?.station);
                  return (
                    <tr key={id}>
                      <th scope="row"><Link className="train-link" href={`/train/${id}`}>{journey.train_name ?? journey.train_no}</Link>
                        <small>{journey.train_name ? `${journey.train_no} · ` : ""}{journey.train_type}</small></th>
                      <td><Status mode={latest?.mode} /></td>
                      <td className={delayTone(latest?.current_delay_min)}>{minutes(latest?.current_delay_min)}</td>
                      <td>{nextStation ?? "—"}</td>
                      <td>{clock(next?.expected_arrival)}{next?.expected_arrival && next.expected_arrival.slice(0, 10) !== time?.slice(0, 10) &&
                        <small>{day(next.expected_arrival)}</small>}</td>
                      <td><button className={`icon-button ${selected === id ? "selected" : ""}`} aria-pressed={selected === id}
                        title="Focus train route on map" aria-label={`Focus train ${journey.train_no} on map`}
                        onClick={() => setSelection(selected === id ? null : { id, replay: start?.sent_at })}>
                        <MapPin size={17} aria-hidden="true" /></button></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : <div className="empty-state">
          <TrainFront size={24} aria-hidden="true" />
          <h3>{start ? "No matching trains" : "No journeys loaded yet"}</h3>
          <p>{start ? "Search by train name or number." : "Start a replay to see trains, forecasts and completed stops here."}</p>
        </div>}
      </section>
    </>
  );
}
