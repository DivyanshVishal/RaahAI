"use client";

import { useEffect, useMemo } from "react";
import Link from "next/link";
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, useMap } from "react-leaflet";
import type { LatLngTuple } from "leaflet";
import type { Journey } from "@/lib/messages";
import type { Progress } from "@/lib/replay-state";
import { delayTone, minutes } from "@/lib/display";

const india: LatLngTuple = [22.6, 79.3];
const tiles = process.env.NEXT_PUBLIC_TILE_URL || "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

function point({ lat, lng }: Journey["stops"][number]): LatLngTuple | null {
  if (lat == null || lng == null || Math.abs(lat) > 90 || Math.abs(lng) > 180) return null;
  return [lat, lng];
}

function FitMap({ points }: { points: LatLngTuple[] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length) map.fitBounds(points, { padding: [29, 29], maxZoom: 8 });
    else map.setView(india, 5);
  }, [map, points]);
  return null;
}

export default function NetworkMap({ journeys, progress, selected }: {
  journeys: Journey[];
  progress: Record<number, Progress>;
  selected: number | null;
}) {
  const visible = useMemo(() => selected == null ? journeys : journeys.filter((j) => j.journey_id === selected), [journeys, selected]);
  const stations = useMemo(() => {
    const unique = new Map<string, { station: string; location: LatLngTuple }>();
    for (const journey of visible) {
      for (const stop of journey.stops) {
        const location = point(stop);
        if (location) unique.set(`${stop.station}:${location.join(",")}`, { station: stop.station, location });
      }
    }
    return [...unique.values()];
  }, [visible]);
  const bounds = useMemo(() => stations.map(({ location }) => location), [stations]);
  return (
    <MapContainer center={india} zoom={5} className="network-map" scrollWheelZoom={false} aria-label="Station and train replay map">
      <FitMap points={bounds} />
      <TileLayer url={tiles} attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
      {selected !== null && visible.map((journey) => (
        <Polyline key={journey.journey_id} positions={journey.stops.flatMap((stop) => {
          const location = point(stop);
          return location ? [location] : [];
        })} pathOptions={{ color: "#b83d18", weight: 3, opacity: 0.75, dashArray: "5 6" }} />
      ))}
      {stations.map(({ station, location }) => (
        <CircleMarker key={`${station}:${location.join(",")}`} center={location} radius={3.6}
          pathOptions={{ color: "#31433f", fillColor: "#fffaf0", fillOpacity: 1, weight: 1.3 }}>
          <Popup>{station}</Popup>
        </CircleMarker>
      ))}
      {visible.map((journey) => {
        const update = progress[journey.journey_id]?.latest;
        if (!update || update.mode === "finished") return null;
        const stopNo = "stop_no" in update.trigger ? update.trigger.stop_no : 0;
        const stop = journey.stops.find((value) => value.stop_no === stopNo);
        const location = stop && point(stop);
        if (!location) return null;
        return (
          <CircleMarker key={journey.journey_id} center={location} radius={8}
            pathOptions={{ color: "#fff", weight: 2.5, fillColor: "#b83d18", fillOpacity: 1 }}>
            <Popup>
              <Link href={`/train/${journey.journey_id}`}>{journey.train_name ?? journey.train_no}</Link><br />
              {update.trigger.station} · <span className={delayTone(update.current_delay_min)}>{minutes(update.current_delay_min)}</span>
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
