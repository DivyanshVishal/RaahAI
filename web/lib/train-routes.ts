import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { unstable_cache } from "next/cache";
import { python, pythonOptions, repo } from "./python";

export type Train = { number: string; name: string | null; file: string };
export type Stop = {
  index: number;
  hasObservedDelay: boolean;
  station: string; platform: string | null; arrival: string | null; departure: string | null;
  actual: string | null; delay: number | null; km: number | null; point: [number, number] | null;
};
export type JourneyRoute = { date: string; dates: string[]; type: string | null; stops: Stop[] };

let catalog: Train[] | undefined;
const run = promisify(execFile);

export async function directory() {
  if (!catalog) {
    const rows: [string, string | null, string][] = JSON.parse(
      await readFile(path.join(repo, "web/data/train-catalog.json"), "utf8")
    );
    catalog = rows.map(([number, name, file]) => ({ number, name, file }));
  }
  return catalog;
}

export function searchTrains(trains: Train[], query: string) {
  const needle = query.trim().toLowerCase();
  if (/^\d+$/.test(needle)) {
    const digits = needle.replace(/^0+(?=\d)/, "");
    const exact = trains.find((train) => train.number === digits);
    return exact ? [exact] : trains.filter((train) => train.number.startsWith(digits));
  }
  const words = needle.split(/\s+/);
  return trains.filter((train) => {
    const text = train.number + " " + (train.name || "").toLowerCase();
    return words.every((word) => text.includes(word));
  });
}

export const loadJourney = unstable_cache(async (file: string, date: string): Promise<JourneyRoute | null> => {
  const args = ["-B", "web/scripts/train_routes.py", "route", file];
  if (date) args.push(date);
  const { stdout } = await run(python, args, { ...pythonOptions, timeout: 45000, maxBuffer: 4 * 1024 * 1024 });
  return JSON.parse(stdout);
}, ["recorded-train-route-v3"], { revalidate: 3600 });
