import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, rmdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { features, python, pythonOptions } from "@/lib/python";
import { manualInput, parsePredictionCsv } from "@/lib/manual-prediction";

const run = promisify(execFile);
export const runtime = "nodejs";

export async function POST(request: Request) {
  const parsed = manualInput.safeParse(await request.json().catch(() => undefined));
  if (!parsed.success) {
    return Response.json({ error: "Enter a train number, valid journey date, and a station if specifying a delay." }, { status: 400 });
  }
  if (!existsSync(path.join(features, "_train_index.csv"))) {
    return Response.json({ error: "The existing feature train index is required for manual predictions." }, { status: 503 });
  }

  const { train, date, station, delay } = parsed.data;
  const directory = await mkdtemp(path.join(tmpdir(), "rail-predict-"));
  const file = path.join(directory, "prediction.csv");
  const args = ["-B", "predict.py", "--train", train, "--date", date, "--csv", file];
  if (station) args.push("--at", station);
  if (delay !== undefined) args.push("--delay", String(delay));
  try {
    await run(python, args, { ...pythonOptions, timeout: 90_000, maxBuffer: 2 * 1024 * 1024 });
    const rows = parsePredictionCsv(await readFile(file, "utf8"), Boolean(station));
    return Response.json({ mode: station ? "live" : "pre_departure", rows });
  } catch (error) {
    const failure = error as Error & { code?: string | number; stderr?: string; killed?: boolean };
    const detail = failure.stderr?.trim();
    if (failure.code === 1 && !failure.killed && detail && !detail.includes("Traceback")) {
      return Response.json({ error: detail }, { status: 422 });
    }
    console.error("Prediction subprocess failed", error);
    return Response.json({ error: failure.killed ? "The Python prediction timed out." : "The Python prediction could not run. Check the configured Python environment." }, { status: 500 });
  } finally {
    await rm(file, { force: true });
    await rmdir(directory);
  }
}
