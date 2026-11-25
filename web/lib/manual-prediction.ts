import { z } from "zod";
import { date } from "./messages";

export const manualInput = z.strictObject({
  train: z.string().trim().regex(/^\d+$/).max(16),
  date,
  station: z.string().trim().toUpperCase().regex(/^[A-Z0-9-]+$/).max(24).optional(),
  delay: z.number().optional(),
}).refine(({ station, delay }) => station !== undefined || delay === undefined, "A delay requires a current station");

export type ManualRow = {
  stop: number;
  station: string;
  sch_arrival: string | null;
  actual_delay: number | null;
  pre_delay: number | null;
  pre_low: number | null;
  pre_high: number | null;
  stops_ahead?: number | null;
  live_delay?: number | null;
  live_low?: number | null;
  live_high?: number | null;
  best_delay?: number | null;
  best_low?: number | null;
  best_high?: number | null;
};

const pre = ["stop", "station", "sch_arrival", "actual_delay", "pre_delay", "pre_low", "pre_high"];
const live = ["stop", "stops_ahead", ...pre.slice(1), "live_delay", "live_low", "live_high",
  "best_delay", "best_low", "best_high"];

export function parsePredictionCsv(csv: string, isLive: boolean): ManualRow[] {
  const [header, ...lines] = csv.trimEnd().split(/\r?\n/);
  const columns = isLive ? live : pre;
  if (header !== columns.join(",")) throw new Error("Unexpected prediction CSV columns");
  return lines.map((line) => {
    const fields = [...line.matchAll(/(?:^|,)(?:"((?:[^"]|"")*)"|([^",]*))/g)]
      .map(([, quoted, plain]) => quoted === undefined ? plain : quoted.replaceAll('""', '"'));
    if (fields.length !== columns.length) throw new Error("Unexpected prediction CSV row");
    const parsed = Object.fromEntries(columns.map((column, index) => {
      const value = fields[index];
      if (column === "station" || column === "sch_arrival") return [column, value || null];
      const number = Number(value);
      return [column, value !== "" && Number.isFinite(number) ? number : null];
    }));
    return parsed as ManualRow;
  });
}
