"use client";

import { useEffect, useState, type FormEvent } from "react";
import { Play } from "lucide-react";
import { DEFAULT_REPLAY_DATE, FIRST_REPLAY_DATE, LAST_REPLAY_DATE, isReplayDate } from "@/lib/replay-dates";
import { useReplay } from "./shell";

type Job = { status: "idle" | "running" | "completed" | "failed"; error?: string };

export function ReplayControls() {
  const { state } = useReplay();
  const [open, setOpen] = useState(false);
  const [job, setJob] = useState<Job>({ status: "idle" });
  const [launching, setLaunching] = useState(false);

  useEffect(() => {
    let active = true;
    async function readJob() {
      try {
        const response = await fetch("/api/simulate", { cache: "no-store" });
        if (active && response.ok) setJob(await response.json());
      } catch {}
    }
    void readJob();
    if (job.status !== "running") return () => { active = false; };
    const timer = setInterval(readJob, 1500);
    return () => { active = false; clearInterval(timer); };
  }, [job.status]);

  async function start(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setLaunching(true);
    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          date: form.get("date"),
          startTime: form.get("startTime"), hours: Number(form.get("hours")),
          speed: Number(form.get("speed")), trains: Number(form.get("trains")),
        }),
      });
      const result = await response.json();
      setJob(response.ok ? result : { status: "failed", error: result.error });
      if (response.ok) setOpen(false);
    } catch {
      setJob({ status: "failed", error: "Unable to reach the local replay service." });
    } finally {
      setLaunching(false);
    }
  }

  const busy = state.phase === "running" || job.status === "running" || launching;
  const lastDate = state.start?.settings.date ?? DEFAULT_REPLAY_DATE;
  return (
    <div className="replay-controls">
      <button className="button-primary" disabled={busy} onClick={() => setOpen(!open)} aria-expanded={open}>
        <Play size={16} fill="currentColor" aria-hidden="true" />
        {busy ? (state.phase === "running" ? "Replay running" : "Starting replay…") : "Start replay"}
      </button>
      {open && (
        <form className="panel replay-popover" onSubmit={start}>
          <h2>Replay settings</h2>
          <p className="muted" id="replay-date-note">August 2026 · IST. The first run for a date may take longer.</p>
          <div className="control-fields">
            <label className="replay-date">Journey date<input name="date" type="date"
              min={FIRST_REPLAY_DATE} max={LAST_REPLAY_DATE}
              defaultValue={isReplayDate(lastDate) ? lastDate : DEFAULT_REPLAY_DATE}
              aria-describedby="replay-date-note" required /></label>
            <label>Start time<input name="startTime" type="time" defaultValue="07:00" required /></label>
            <label>Hours<input name="hours" type="number" min="0.1" max="24" step="0.1" defaultValue="3" required /></label>
            <label>Speed (×)<input name="speed" type="number" min="1" max="3600" defaultValue="120" required /></label>
            <label>Trains<input name="trains" type="number" min="1" max="100" defaultValue="30" required /></label>
          </div>
          <button className="button-primary" disabled={busy} type="submit">Run replay</button>
        </form>
      )}
      {job.error && <p role="alert" className="control-error">{job.error}</p>}
    </div>
  );
}
