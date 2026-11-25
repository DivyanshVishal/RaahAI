import type { Update } from "./messages";

export function clock(value: string | null | undefined) {
  return value?.slice(11, 16) || "—";
}

export function day(value: string | null | undefined) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", year: "numeric", timeZone: "Asia/Kolkata" })
    .format(new Date(`${value.slice(0, 10)}T00:00:00+05:30`));
}

export function minutes(value: number | null | undefined, suffix = true) {
  if (value == null) return "—";
  if (Math.abs(value) < 0.05) return suffix ? "0 min" : "0";
  return `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(1).replace(/\.0$/, "")}${suffix ? " min" : ""}`;
}

export function interval(low: number | null | undefined, high: number | null | undefined) {
  return low == null || high == null ? "—" : `${minutes(low, false)} to ${minutes(high, false)} min`;
}

export function delayTone(value: number | null | undefined) {
  return value == null ? "" : value > 0.05 ? "delay-late" : "delay-clear";
}

const modes = {
  pre_departure: "Pre-departure",
  live: "Running",
  finished: "Finished",
  no_live_data: "No live forecast",
  waiting: "Awaiting replay",
};

export function Status({ mode }: { mode?: Update["mode"] }) {
  const key = mode ?? "waiting";
  return <span className={`status status-${key}`}>{modes[key]}</span>;
}
