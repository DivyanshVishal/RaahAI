import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, realpath, symlink } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { date } from "@/lib/messages";
import { features, python, pythonOptions, repo } from "@/lib/python";
import { DEFAULT_REPLAY_DATE, isReplayDate } from "@/lib/replay-dates";
import { getState } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const options = z.strictObject({
  date: date.default(DEFAULT_REPLAY_DATE),
  startTime: z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/).default("07:00"),
  hours: z.number().positive().max(24).default(3),
  speed: z.number().min(1).max(3600).default(120),
  trains: z.int().min(1).max(100).default(30),
});
type Job = { status: "idle" | "running" | "completed" | "failed"; error?: string };
const host = globalThis as typeof globalThis & { railReplayJob?: Job };

async function linkReplayData(directory: string, source: string) {
  await mkdir(directory, { recursive: true });
  const link = path.join(directory, "data");
  const target = await realpath(source);
  try {
    await symlink(target, link, process.platform === "win32" ? "junction" : "dir");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST" ||
      path.relative(target, await realpath(link)) !== "") throw error;
  }
}

async function replayDataArgs(directory: string, date: string) {
  if (existsSync(path.join(features, `_sim_journeys_${date}.csv`))) return [];

  // The simulator writes its journey index to --features and uses the same
  // relative file names under --clean to look up train names.
  const replayFeatures = path.join(directory, "features");
  const replayClean = path.join(directory, "clean");
  await Promise.all([
    linkReplayData(replayFeatures, features),
    linkReplayData(replayClean, path.join(repo, "processed", "clean_data", "year=2026")),
  ]);
  return ["--features", replayFeatures, "--clean", replayClean];
}

export function GET() {
  return Response.json(host.railReplayJob ?? { status: "idle" }, { headers: { "Cache-Control": "no-store" } });
}

export async function POST(request: Request) {
  const parsed = options.safeParse(await request.json().catch(() => undefined));
  if (!parsed.success) return Response.json({ error: "Invalid replay options." }, { status: 400 });
  const { date, startTime, hours, speed, trains } = parsed.data;
  if (!isReplayDate(date)) {
    return Response.json({ error: "Choose a replay date in August 2026." }, { status: 422 });
  }
  if (host.railReplayJob?.status === "running" || getState().phase === "running") {
    return Response.json({ error: "A replay is already in progress." }, { status: 409 });
  }
  host.railReplayJob = { status: "running" };

  const directory = path.resolve(".replay");
  try {
    await mkdir(directory, { recursive: true });
    const port = process.env.REPLAY_PORT || new URL(request.url).port || "3000";
    const dataArgs = await replayDataArgs(directory, date);
    const args = [
      "-B", "-u", "simulate_live.py", "--date", date, "--start-time", startTime, "--hours", String(hours),
      "--speed", String(speed), "--trains", String(trains),
      "--url", `http://localhost:${port}/api/predictions`,
      "--log", path.join(directory, "sim_log.jsonl"), "--outbox", path.join(directory, "sim_outbox.jsonl"),
      ...dataArgs,
    ];
    const runAt = Date.now();
    execFile(python, args, { ...pythonOptions, maxBuffer: 2 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error || (!getState().start || !/delivered [1-9]\d*, failed 0\b/.test(stdout))) {
        console.error("Replay subprocess failed", { error, stderr, stdout: stdout.slice(-1500), elapsedMs: Date.now() - runAt });
        host.railReplayJob = { status: "failed", error: stderr.trim().split(/\r?\n/).at(-1) ||
          "The replay did not deliver all messages. Check the local Python configuration and replay log." };
        return;
      }
      host.railReplayJob = { status: "completed" };
    });
    return Response.json({ status: "running" }, { status: 202 });
  } catch (error) {
    console.error("Replay launch failed", error);
    host.railReplayJob = { status: "failed", error: "The replay could not start." };
    return Response.json(host.railReplayJob, { status: 500 });
  }
}
