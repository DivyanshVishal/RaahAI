"use client";

import { useEffect, useMemo } from "react";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { LatLngTuple } from "leaflet";
import type { Stop } from "@/lib/train-routes";

function Frame({ points, focus }: { points: LatLngTuple[]; focus: LatLngTuple | null }) {
  const map = useMap();
  useEffect(() => {
    if (focus) map.setView(focus, Math.max(map.getZoom(), 8));
    else if (points.length) map.fitBounds(points, { padding: [42, 40], maxZoom: 10, animate: false });
  }, [map, points, focus]);
  return null;
}

export default function RouteMap({ stops, selected, onSelect }: {
  stops: Stop[]; selected: number | null; onSelect: (i: number) => void;
}) {
  const points = useMemo(() => stops.flatMap((s) => s.point ? [s.point] : []), [stops]);
  if (!points.length) return <div className="atlas-map map-placeholder">Coordinates are unavailable for this journey.</div>;
  return <MapContainer className="atlas-map" center={points[0]} zoom={6} scrollWheelZoom={false} aria-label="Recorded train route">
    <Frame points={points} focus={selected === null ? null : stops[selected].point} />
    <TileLayer url={process.env.NEXT_PUBLIC_TILE_URL || "https://tile.openstreetmap.org/{z}/{x}/{y}.png"}
      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
    {points.length > 1 && <Polyline positions={points} pathOptions={{ color: "#ac401f", weight: 4, opacity: .85 }} />}
    {stops.map((stop, i) => {
      if (!stop.point) return null;
      const end = i === 0 || i === stops.length - 1;
      const active = selected === i;
      return <CircleMarker key={i} center={stop.point} radius={active ? 9 : end ? 7 : 5}
        eventHandlers={{ click: () => onSelect(i) }}
        pathOptions={{ color: "#783118", weight: 2, fillColor: active || end ? "#d55b30" : "#fffaf0", fillOpacity: 1 }}>
        <Tooltip key={String(active)} permanent={active || (end && selected === null)} direction="top" offset={[0, -6]}>{stop.station}</Tooltip>
      </CircleMarker>;
    })}
  </MapContainer>;
}
