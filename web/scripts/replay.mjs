import { readFile } from "node:fs/promises";
import { setTimeout } from "node:timers/promises";

const file = process.argv[2] ?? "../sim_log.jsonl";
const interval = Number(process.argv[3] ?? 700);
const base = process.argv[4] ?? "http://localhost:3000";
if (!Number.isFinite(interval) || interval < 0) throw new Error("Pass a nonnegative delay in milliseconds.");

const messages = (await readFile(file, "utf8")).split(/\r?\n/).filter((line) => line.trim());
for (const [index, line] of messages.entries()) {
  if (index) await setTimeout(interval);
  const message = JSON.parse(line);
  const response = await fetch(new URL("/api/predictions", base), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: line,
  });
  if (!response.ok) throw new Error(`Line ${index + 1}: HTTP ${response.status} ${await response.text()}`);
  console.log(message.batch_no ? `batch ${message.batch_no} · ${message.stats.checks} checks` : message.type);
}
