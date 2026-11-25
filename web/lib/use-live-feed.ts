"use client";

import { useEffect, useState } from "react";

type Observation = {
  type: "arrival" | "departure";
  station: string;
  stop_index: number;
  actual_at: string;
  delay_min: number | null;
};
type Forecast = {
  stop_index: number;
  station: string;
  from_station: string;
  from_type: "arrival" | "departure";
  delay: number;
  low: number;
  high: number;
};
type Score = Forecast & { actual: number; pre: number };
type Batch = {
  cursor: number;
  run: string;
  complete: boolean;
  simulated_at: string;
  speed: number;
  max_gap_seconds?: number;
  events: Observation[];
  forecasts?: Forecast[];
  scores?: Score[];
};

export function useLiveFeed(enabled: boolean, train: string, date: string, restart: number) {
  const [state, setState] = useState<{ events: Observation[]; batch: Batch | null; message: string }>({
    events: [], batch: null, message: "",
  });

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let events: Observation[] = [];
    let cursor = 0;
    let run = "";
    const session = crypto.randomUUID();
    setState({ events, batch: null, message: "Connecting to the simulated feed…" });

    async function poll() {
      try {
        const query = new URLSearchParams({ train_no: train, journey_date: date, after: String(cursor), run, session });
        const result = await fetch("/api/live-feed?" + query, { signal: controller.signal });
        if (controller.signal.aborted) return;
        if (result.status === 409) {
          cursor = 0;
          run = "";
          events = [];
          setState({ events, batch: null, message: "Reconnecting to the restarted feed…" });
        } else if (result.ok) {
          const batch: Batch = await result.json();
          if (controller.signal.aborted) return;
          cursor = batch.cursor;
          run = batch.run;
          events = [...events, ...batch.events];
          const message = !batch.forecasts ? "Station reports available. This journey has no model updates."
            : batch.complete ? "Simulation complete." : "Receiving simulated station reports.";
          setState({ events, batch, message });
          if (batch.complete) return;
        } else {
          const resultBody = await result.json();
          if (controller.signal.aborted) return;
          setState((previous) => ({ ...previous, message: resultBody.error }));
          if (result.status === 400 || result.status === 404) return;
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        setState((previous) => ({ ...previous, message: error instanceof Error ? error.message : "Feed unavailable." }));
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 1000);
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [enabled, train, date, restart]);

  return state;
}
